from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

from tqdm import tqdm

from finetuning_buckets.inference import chat
from finetuning_buckets.inference.safety_eval.evaluator import common_eval_template
from finetuning_buckets.inference.safety_eval.keyword_eval import KeyWordEvaluator


ENV_PATH = Path(".env")
DEFAULT_HARMFUL_DATA_PATH = "finetuning_buckets/datasets/data/safety_bench/Harmful-HEx-PHI.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine adaptive harmful assistant prefixes through Tinker. The script seeds the "
            "target model with short harmful prefixes, lets it continue, and keeps prefixes "
            "that induce a non-refusal verification continuation."
        )
    )
    parser.add_argument("--model_name", default=None, help="Tinker base model name.")
    parser.add_argument("--model_path", default=None, help="Optional Tinker checkpoint path.")
    parser.add_argument("--base_url", default=None, help="Optional Tinker API base URL.")
    parser.add_argument("--prompt_style", default="llama3", help="Repo prompt formatter style.")
    parser.add_argument(
        "--eval_template",
        default="null",
        choices=sorted(common_eval_template.keys()),
        help="Template key from common_eval_template.",
    )
    parser.add_argument(
        "--harmful_data_path",
        default=DEFAULT_HARMFUL_DATA_PATH,
        help="JSONL with HEx-PHI harmful prompt/answer pairs.",
    )
    parser.add_argument(
        "--seed_prefix_path",
        default=None,
        help=(
            "Optional JSONL of prior mined prefixes. When set, records must contain "
            "`instruction` and a prefix field, and those prefixes become the mining seeds."
        ),
    )
    parser.add_argument(
        "--seed_prefix_field",
        default="prefix",
        help="Field to read from --seed_prefix_path as the seed text.",
    )
    parser.add_argument(
        "--save_path",
        default="logs/adaptive_prefixes/mined_prefixes_tinker.jsonl",
        help="Where kept adaptive prefixes are written as JSONL.",
    )
    parser.add_argument(
        "--metadata_path",
        default=None,
        help="Optional metadata JSON path. Defaults to save_path with .metadata.json suffix.",
    )
    parser.add_argument(
        "--save_all_candidates_path",
        default=None,
        help="Optional JSONL path for every attempted candidate, including rejected ones.",
    )
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum prompts to mine.")
    parser.add_argument(
        "--seed_prefix_tokens",
        default="5,10,20",
        help="Comma-separated seed prefix lengths to try. Use `full` to use the full seed text.",
    )
    parser.add_argument("--samples_per_seed", type=int, default=1)
    parser.add_argument(
        "--max_kept_per_prompt",
        type=int,
        default=1,
        help="Stop mining a prompt after this many successful prefixes; <=0 keeps all successes.",
    )
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument(
        "--continuation_tokens",
        type=int,
        default=20,
        help="Tokens generated after the harmful seed to form an adaptive candidate prefix.",
    )
    parser.add_argument(
        "--verification_tokens",
        type=int,
        default=256,
        help="Tokens generated from the candidate prefix to check whether it induces unsafe continuation.",
    )
    parser.add_argument(
        "--max_prefix_tokens",
        type=int,
        default=80,
        help="Maximum total candidate prefix length after seed plus mined continuation; <=0 disables.",
    )
    parser.add_argument("--mining_temperature", type=float, default=1.0)
    parser.add_argument("--mining_top_p", type=float, default=0.95)
    parser.add_argument("--mining_top_k", type=int, default=None)
    parser.add_argument("--verification_temperature", type=float, default=0.7)
    parser.add_argument("--verification_top_p", type=float, default=0.9)
    parser.add_argument("--verification_top_k", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--evaluator", choices=["key_word", "none"], default="key_word")
    parser.add_argument(
        "--keep_refusal_candidates",
        action="store_true",
        help="If set, verify mined continuations even when they look like refusals.",
    )
    return parser.parse_args()


def load_dotenv_if_available() -> None:
    if not ENV_PATH.exists():
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(ENV_PATH, override=False)


def _require_tinker():
    try:
        import tinker
        from tinker import types
    except ImportError as exc:
        raise SystemExit(
            "Failed to import Tinker dependencies.\n"
            f"Underlying error: {type(exc).__name__}: {exc}\n\n"
            "Install them in your active environment, e.g.:\n"
            "  pip install -U tinker tinker-cookbook"
        ) from exc

    return tinker, types


def parse_seed_specs(value: str) -> list[int | None]:
    lengths = []
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if piece.lower() == "full":
            lengths.append(None)
            continue
        length = int(piece)
        if length <= 0:
            raise ValueError("seed_prefix_tokens must contain positive integers or `full`")
        lengths.append(length)
    if not lengths:
        raise ValueError("seed_prefix_tokens must contain at least one positive integer or `full`")
    return lengths


def load_harmful_examples(path: str) -> list[dict[str, Any]]:
    examples = []
    with open(path, "r") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, list) or len(item) < 2:
                raise ValueError(f"Expected list-style harmful example at line {idx + 1}")
            user_message, assistant_message = item[0], item[1]
            examples.append(
                {
                    "id": idx,
                    "instruction": user_message["content"],
                    "seed_text": assistant_message["content"],
                    "seed_source": "harmful_hex_phi",
                    "category": user_message.get("category"),
                    "parent_record": None,
                }
            )
    return examples


def load_seed_prefix_examples(path: str, seed_prefix_field: str) -> list[dict[str, Any]]:
    examples = []
    with open(path, "r") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            if "instruction" not in record or seed_prefix_field not in record:
                raise ValueError(
                    f"Record on line {idx + 1} of {path} must contain "
                    f"`instruction` and `{seed_prefix_field}` fields"
                )
            examples.append(
                {
                    "id": record.get("id", idx),
                    "instruction": record["instruction"],
                    "seed_text": record[seed_prefix_field],
                    "seed_source": path,
                    "category": record.get("category"),
                    "parent_record": record,
                }
            )
    return examples


def build_messages(system_prompt: str | None, input_template: str | None, instruction: str, assistant_prefix: str):
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    user_content = input_template % instruction if input_template is not None else instruction
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": assistant_prefix})
    return messages


def first_k_decoded_tokens(tokenizer, text: str, k: int | None) -> str:
    token_ids = tokenizer.encode(text)[:k]
    return tokenizer.decode(token_ids)


def truncate_decoded_tokens(tokenizer, text: str, max_tokens: int) -> tuple[str, int]:
    token_ids = tokenizer.encode(text)
    if max_tokens > 0:
        token_ids = token_ids[:max_tokens]
    return tokenizer.decode(token_ids), len(token_ids)


def is_jailbroken(text: str, evaluator: str) -> bool:
    if evaluator == "none":
        return True
    if evaluator == "key_word":
        return KeyWordEvaluator.is_jailbroken(text)
    raise ValueError(f"Evaluator {evaluator} not maintained")


def build_sampling_params(types, max_tokens: int, temperature: float, top_p: float, top_k: int | None):
    kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if top_k is not None:
        kwargs["top_k"] = top_k
    return types.SamplingParams(**kwargs)


async def sample_text(
    sampling_client,
    types,
    formatter: chat.Chat,
    tokenizer,
    messages: list[dict],
    sampling_params,
    semaphore: asyncio.Semaphore,
) -> str:
    async with semaphore:
        item = formatter.validate_conversation(messages)
        prompt_text = formatter.string_formatter({"messages": item})["text"]
        prompt = types.ModelInput.from_ints(tokenizer.encode(prompt_text))
        sampled = await sampling_client.sample_async(
            prompt=prompt,
            sampling_params=sampling_params,
            num_samples=1,
        )
        return tokenizer.decode(list(sampled.sequences[0].tokens))


async def mine_prompt(
    example: dict[str, Any],
    args: argparse.Namespace,
    types,
    sampling_client,
    formatter: chat.Chat,
    tokenizer,
    mining_sampling_params,
    verification_sampling_params,
    semaphore: asyncio.Semaphore,
    system_prompt: str | None,
    input_template: str | None,
    seed_specs: list[int | None],
    mined_against: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept_records = []
    all_records = []

    for seed_tokens in seed_specs:
        seed_prefix = first_k_decoded_tokens(tokenizer, example["seed_text"], seed_tokens)
        if not seed_prefix.strip():
            continue
        seed_token_count = len(tokenizer.encode(seed_prefix))
        seed_spec = "full" if seed_tokens is None else seed_tokens

        for sample_idx in range(args.samples_per_seed):
            if args.max_kept_per_prompt > 0 and len(kept_records) >= args.max_kept_per_prompt:
                return kept_records, all_records

            mining_messages = build_messages(
                system_prompt,
                input_template,
                example["instruction"],
                seed_prefix,
            )
            mined_continuation = await sample_text(
                sampling_client=sampling_client,
                types=types,
                formatter=formatter,
                tokenizer=tokenizer,
                messages=mining_messages,
                sampling_params=mining_sampling_params,
                semaphore=semaphore,
            )

            if not args.keep_refusal_candidates and not is_jailbroken(mined_continuation, "key_word"):
                all_records.append(
                    {
                        "id": example["id"],
                        "instruction": example["instruction"],
                        "category": example["category"],
                        "seed_tokens": seed_spec,
                        "seed_token_count": seed_token_count,
                        "seed_source": example["seed_source"],
                        "parent_prefix_tokens": (example["parent_record"] or {}).get("prefix_tokens"),
                        "sample_idx": sample_idx,
                        "harmful_seed": seed_prefix,
                        "seed_prefix": seed_prefix,
                        "mined_continuation": mined_continuation,
                        "candidate_prefix": seed_prefix + mined_continuation,
                        "mined_against": mined_against,
                        "kept": False,
                        "reject_reason": "refusal_like_mined_continuation",
                    }
                )
                continue

            candidate_prefix, candidate_prefix_tokens = truncate_decoded_tokens(
                tokenizer,
                seed_prefix + mined_continuation,
                args.max_prefix_tokens,
            )
            verification_messages = build_messages(
                system_prompt,
                input_template,
                example["instruction"],
                candidate_prefix,
            )
            verification_continuation = await sample_text(
                sampling_client=sampling_client,
                types=types,
                formatter=formatter,
                tokenizer=tokenizer,
                messages=verification_messages,
                sampling_params=verification_sampling_params,
                semaphore=semaphore,
            )

            keep = is_jailbroken(verification_continuation, args.evaluator)
            record = {
                "id": example["id"],
                "instruction": example["instruction"],
                "category": example["category"],
                "prefix": candidate_prefix,
                "prefix_tokens": candidate_prefix_tokens,
                "prefix_source": "seed_prefix_plus_tinker_target_model_continuation",
                "seed_tokens": seed_spec,
                "seed_token_count": seed_token_count,
                "seed_source": example["seed_source"],
                "parent_prefix_tokens": (example["parent_record"] or {}).get("prefix_tokens"),
                "parent_prefix_source": (example["parent_record"] or {}).get("prefix_source"),
                "sample_idx": sample_idx,
                "harmful_seed": seed_prefix,
                "seed_prefix": seed_prefix,
                "mined_continuation": mined_continuation,
                "verification_continuation": verification_continuation,
                "mined_against": mined_against,
                "evaluator": args.evaluator,
                "kept": keep,
                "reject_reason": None if keep else "safe_or_refusal_like_verification",
            }
            all_records.append(record)
            if keep:
                kept_records.append(record)

    return kept_records, all_records


async def mine_prefixes(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv_if_available()
    tinker, types = _require_tinker()

    if args.model_name is None and args.model_path is None:
        raise SystemExit("Provide either --model_name or --model_path.")
    if args.model_name is not None and args.model_path is not None:
        raise SystemExit("Use only one of --model_name or --model_path.")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set. Put it in `.env` or export it before running.")
    if args.samples_per_seed <= 0:
        raise ValueError("samples_per_seed must be positive")
    if args.continuation_tokens <= 0:
        raise ValueError("continuation_tokens must be positive")
    if args.verification_tokens <= 0:
        raise ValueError("verification_tokens must be positive")
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")

    random.seed(args.random_seed)
    seed_specs = parse_seed_specs(args.seed_prefix_tokens)

    service_client = tinker.ServiceClient(base_url=args.base_url)
    sampling_client_kwargs = {}
    if args.model_path is not None:
        sampling_client_kwargs["model_path"] = args.model_path
    else:
        sampling_client_kwargs["base_model"] = args.model_name
    sampling_client = await service_client.create_sampling_client_async(**sampling_client_kwargs)
    tokenizer = sampling_client.get_tokenizer()

    eval_template = common_eval_template[args.eval_template]
    system_prompt = eval_template["system_prompt"]
    input_template = eval_template["input_template"]
    formatter = chat.Chat(
        model=None,
        prompt_style=args.prompt_style,
        tokenizer=tokenizer,
        init_system_prompt=system_prompt,
    )

    mining_sampling_params = build_sampling_params(
        types,
        max_tokens=args.continuation_tokens,
        temperature=args.mining_temperature,
        top_p=args.mining_top_p,
        top_k=args.mining_top_k,
    )
    verification_sampling_params = build_sampling_params(
        types,
        max_tokens=args.verification_tokens,
        temperature=args.verification_temperature,
        top_p=args.verification_top_p,
        top_k=args.verification_top_k,
    )

    if args.seed_prefix_path is not None:
        examples = load_seed_prefix_examples(args.seed_prefix_path, args.seed_prefix_field)
        seed_data_path = args.seed_prefix_path
        seed_data_type = "prior_mined_prefixes"
    else:
        examples = load_harmful_examples(args.harmful_data_path)
        seed_data_path = args.harmful_data_path
        seed_data_type = "harmful_hex_phi"
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    semaphore = asyncio.Semaphore(args.concurrency)
    mined_against = args.model_path or args.model_name
    tasks = [
        asyncio.create_task(
            mine_prompt(
                example=example,
                args=args,
                types=types,
                sampling_client=sampling_client,
                formatter=formatter,
                tokenizer=tokenizer,
                mining_sampling_params=mining_sampling_params,
                verification_sampling_params=verification_sampling_params,
                semaphore=semaphore,
                system_prompt=system_prompt,
                input_template=input_template,
                seed_specs=seed_specs,
                mined_against=mined_against,
            )
        )
        for example in examples
    ]

    kept_records = []
    all_records = []
    progress = tqdm(total=len(tasks), desc="Mining adaptive prefixes via Tinker")
    for future in asyncio.as_completed(tasks):
        prompt_kept, prompt_all = await future
        kept_records.extend(prompt_kept)
        all_records.extend(prompt_all)
        progress.update(1)
    progress.close()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w") as handle:
        for record in kept_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.save_all_candidates_path is not None:
        all_path = Path(args.save_all_candidates_path)
        all_path.parent.mkdir(parents=True, exist_ok=True)
        with all_path.open("w") as handle:
            for record in all_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    attempted = len(all_records)
    rejected_refusal_candidate = sum(
        record.get("reject_reason") == "refusal_like_mined_continuation" for record in all_records
    )
    rejected_verification = sum(
        record.get("reject_reason") == "safe_or_refusal_like_verification" for record in all_records
    )

    metadata = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "prompt_style": args.prompt_style,
        "eval_template": args.eval_template,
        "harmful_data_path": args.harmful_data_path,
        "seed_prefix_path": args.seed_prefix_path,
        "seed_prefix_field": args.seed_prefix_field,
        "seed_data_path": seed_data_path,
        "seed_data_type": seed_data_type,
        "save_path": args.save_path,
        "save_all_candidates_path": args.save_all_candidates_path,
        "num_examples": len(examples),
        "attempted_candidates": attempted,
        "kept_prefixes": len(kept_records),
        "rejected_refusal_candidate": rejected_refusal_candidate,
        "rejected_verification": rejected_verification,
        "seed_prefix_tokens": ["full" if item is None else item for item in seed_specs],
        "samples_per_seed": args.samples_per_seed,
        "max_kept_per_prompt": args.max_kept_per_prompt,
        "continuation_tokens": args.continuation_tokens,
        "verification_tokens": args.verification_tokens,
        "max_prefix_tokens": args.max_prefix_tokens,
        "mining_temperature": args.mining_temperature,
        "mining_top_p": args.mining_top_p,
        "mining_top_k": args.mining_top_k,
        "verification_temperature": args.verification_temperature,
        "verification_top_p": args.verification_top_p,
        "verification_top_k": args.verification_top_k,
        "concurrency": args.concurrency,
        "evaluator": args.evaluator,
    }
    metadata_path = Path(args.metadata_path) if args.metadata_path else save_path.with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)

    return metadata


def main() -> None:
    args = parse_args()
    metadata = asyncio.run(mine_prefixes(args))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
