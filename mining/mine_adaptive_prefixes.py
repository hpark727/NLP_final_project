from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from finetuning_buckets.inference import chat
from finetuning_buckets.inference.safety_eval.evaluator import common_eval_template
from finetuning_buckets.inference.safety_eval.keyword_eval import KeyWordEvaluator
from finetuning_buckets.models import get_model


DEFAULT_HARMFUL_DATA_PATH = "finetuning_buckets/datasets/data/safety_bench/Harmful-HEx-PHI.jsonl"


@dataclass
class ScriptArguments:
    prompt_style: str = field(default="llama2", metadata={"help": "Prompt style used by the chat formatter."})
    model_family: str = field(default="llama2", metadata={"help": "Model family used by get_model."})
    eval_template: str = field(default="null", metadata={"help": "Template key from common_eval_template."})
    harmful_data_path: str = field(
        default=DEFAULT_HARMFUL_DATA_PATH,
        metadata={"help": "JSONL with HEx-PHI harmful prompt/answer pairs."},
    )
    save_path: str = field(
        default="logs/adaptive_prefixes/mined_prefixes.jsonl",
        metadata={"help": "Where kept adaptive prefixes are written as JSONL."},
    )
    metadata_path: str | None = field(
        default=None,
        metadata={"help": "Optional metadata JSON path. Defaults to save_path with .metadata.json suffix."},
    )

    max_examples: int = field(default=-1, metadata={"help": "Maximum prompts to mine; <=0 uses all."})
    seed_prefix_tokens: str = field(
        default="5,10,20",
        metadata={"help": "Comma-separated harmful seed prefix lengths to try."},
    )
    samples_per_seed: int = field(default=1, metadata={"help": "Number of mining continuations per seed length."})
    max_kept_per_prompt: int = field(
        default=1,
        metadata={"help": "Stop mining a prompt after this many successful prefixes; <=0 keeps all successes."},
    )
    random_seed: int = field(default=0, metadata={"help": "Python and torch random seed."})

    continuation_tokens: int = field(
        default=20,
        metadata={"help": "Tokens generated after the harmful seed to form an adaptive candidate prefix."},
    )
    verification_tokens: int = field(
        default=256,
        metadata={"help": "Tokens generated from the candidate prefix to check whether it induces unsafe continuation."},
    )
    max_prefix_tokens: int = field(
        default=80,
        metadata={"help": "Maximum total candidate prefix length after seed plus mined continuation; <=0 disables."},
    )

    mining_temperature: float = field(default=1.0, metadata={"help": "Temperature for candidate-prefix mining."})
    mining_top_p: float = field(default=0.95, metadata={"help": "Top-p for candidate-prefix mining."})
    mining_top_k: int = field(default=50, metadata={"help": "Top-k for candidate-prefix mining."})
    verification_temperature: float = field(default=0.7, metadata={"help": "Temperature for verification generation."})
    verification_top_p: float = field(default=0.9, metadata={"help": "Top-p for verification generation."})
    verification_top_k: int = field(default=50, metadata={"help": "Top-k for verification generation."})
    repetition_penalty: float = field(default=1.0, metadata={"help": "Repetition penalty for both generation phases."})

    evaluator: str = field(
        default="key_word",
        metadata={"help": "Prefix keeper heuristic. Currently supports key_word or none."},
    )
    keep_refusal_candidates: bool = field(
        default=False,
        metadata={"help": "If false, skip candidates whose mined continuation already looks like a refusal."},
    )
    save_all_candidates_path: str | None = field(
        default=None,
        metadata={"help": "Optional JSONL path for every attempted candidate, including rejected ones."},
    )


def parse_int_list(value: str) -> list[int]:
    lengths = []
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        length = int(piece)
        if length <= 0:
            raise ValueError("seed_prefix_tokens must contain positive integers")
        lengths.append(length)
    if not lengths:
        raise ValueError("seed_prefix_tokens must contain at least one positive integer")
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
                    "harmful": assistant_message["content"],
                    "category": user_message.get("category"),
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


def first_k_decoded_tokens(tokenizer, text: str, k: int) -> str:
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:k]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def truncate_decoded_tokens(tokenizer, text: str, max_tokens: int) -> tuple[str, int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if max_tokens > 0:
        token_ids = token_ids[:max_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True), len(token_ids)


def is_jailbroken(text: str, evaluator: str) -> bool:
    if evaluator == "none":
        return True
    if evaluator == "key_word":
        return KeyWordEvaluator.is_jailbroken(text)
    raise ValueError(f"Evaluator {evaluator} not maintained")


def write_jsonl_record(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def main() -> None:
    parser = HfArgumentParser((ScriptArguments, ModelConfig))
    args, model_config = parser.parse_args_into_dataclasses()

    if model_config.model_name_or_path is None:
        raise ValueError("Pass --model_name_or_path for the model to mine against.")
    if args.eval_template not in common_eval_template:
        raise ValueError(f"eval_template {args.eval_template} not maintained")
    if args.samples_per_seed <= 0:
        raise ValueError("samples_per_seed must be positive")
    if args.continuation_tokens <= 0:
        raise ValueError("continuation_tokens must be positive")
    if args.verification_tokens <= 0:
        raise ValueError("verification_tokens must be positive")

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    seed_lengths = parse_int_list(args.seed_prefix_tokens)

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model, tokenizer = get_model.get_model(
        model_config.model_name_or_path,
        model_kwargs,
        model_family=args.model_family,
        padding_side="left",
    )
    model.eval()
    if not quantization_config and torch.cuda.is_available():
        model.to("cuda")

    eval_template = common_eval_template[args.eval_template]
    system_prompt = eval_template["system_prompt"]
    input_template = eval_template["input_template"]
    generator = chat.Chat(model=model, prompt_style=args.prompt_style, tokenizer=tokenizer, init_system_prompt=system_prompt)

    examples = load_harmful_examples(args.harmful_data_path)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata_path) if args.metadata_path else save_path.with_suffix(".metadata.json")
    all_candidates_path = Path(args.save_all_candidates_path) if args.save_all_candidates_path else None
    if all_candidates_path is not None:
        all_candidates_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    attempted = 0
    rejected_refusal_candidate = 0
    rejected_verification = 0

    with open(save_path, "w") as kept_handle:
        all_handle = open(all_candidates_path, "w") if all_candidates_path is not None else None
        try:
            for example in tqdm(examples, desc="Mining adaptive prefixes"):
                kept_for_prompt = 0
                for seed_tokens in seed_lengths:
                    harmful_seed = first_k_decoded_tokens(tokenizer, example["harmful"], seed_tokens)
                    if not harmful_seed.strip():
                        continue

                    for sample_idx in range(args.samples_per_seed):
                        if args.max_kept_per_prompt > 0 and kept_for_prompt >= args.max_kept_per_prompt:
                            break

                        attempted += 1
                        mining_messages = build_messages(
                            system_prompt,
                            input_template,
                            example["instruction"],
                            harmful_seed,
                        )
                        with torch.inference_mode():
                            mined_continuation, _ = generator.generate_one_shot(
                                mining_messages,
                                max_new_tokens=args.continuation_tokens,
                                do_sample=True,
                                top_p=args.mining_top_p,
                                temperature=args.mining_temperature,
                                top_k=args.mining_top_k,
                                repetition_penalty=args.repetition_penalty,
                            )

                        if not args.keep_refusal_candidates and not is_jailbroken(mined_continuation, "key_word"):
                            rejected_refusal_candidate += 1
                            candidate_record = {
                                "id": example["id"],
                                "instruction": example["instruction"],
                                "category": example["category"],
                                "seed_tokens": seed_tokens,
                                "sample_idx": sample_idx,
                                "harmful_seed": harmful_seed,
                                "mined_continuation": mined_continuation,
                                "candidate_prefix": harmful_seed + mined_continuation,
                                "kept": False,
                                "reject_reason": "refusal_like_mined_continuation",
                            }
                            if all_handle is not None:
                                write_jsonl_record(all_handle, candidate_record)
                            continue

                        candidate_prefix, candidate_prefix_tokens = truncate_decoded_tokens(
                            tokenizer,
                            harmful_seed + mined_continuation,
                            args.max_prefix_tokens,
                        )
                        verification_messages = build_messages(
                            system_prompt,
                            input_template,
                            example["instruction"],
                            candidate_prefix,
                        )
                        with torch.inference_mode():
                            verification_continuation, _ = generator.generate_one_shot(
                                verification_messages,
                                max_new_tokens=args.verification_tokens,
                                do_sample=True,
                                top_p=args.verification_top_p,
                                temperature=args.verification_temperature,
                                top_k=args.verification_top_k,
                                repetition_penalty=args.repetition_penalty,
                            )

                        keep = is_jailbroken(verification_continuation, args.evaluator)
                        if keep:
                            kept += 1
                            kept_for_prompt += 1
                        else:
                            rejected_verification += 1

                        record = {
                            "id": example["id"],
                            "instruction": example["instruction"],
                            "category": example["category"],
                            "prefix": candidate_prefix,
                            "prefix_tokens": candidate_prefix_tokens,
                            "prefix_source": "harmful_seed_plus_target_model_continuation",
                            "seed_tokens": seed_tokens,
                            "sample_idx": sample_idx,
                            "harmful_seed": harmful_seed,
                            "mined_continuation": mined_continuation,
                            "verification_continuation": verification_continuation,
                            "mined_against": model_config.model_name_or_path,
                            "evaluator": args.evaluator,
                            "kept": keep,
                            "reject_reason": None if keep else "safe_or_refusal_like_verification",
                        }
                        if keep:
                            write_jsonl_record(kept_handle, record)
                        if all_handle is not None:
                            write_jsonl_record(all_handle, record)

                    if args.max_kept_per_prompt > 0 and kept_for_prompt >= args.max_kept_per_prompt:
                        break
        finally:
            if all_handle is not None:
                all_handle.close()

    metadata = {
        "script_args": asdict(args),
        "model_name_or_path": model_config.model_name_or_path,
        "model_family": args.model_family,
        "prompt_style": args.prompt_style,
        "num_examples": len(examples),
        "attempted_candidates": attempted,
        "kept_prefixes": kept,
        "rejected_refusal_candidate": rejected_refusal_candidate,
        "rejected_verification": rejected_verification,
        "seed_prefix_tokens": seed_lengths,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Attempted candidates: {attempted}")
    print(f"Kept adaptive prefixes: {kept}")
    print(f"Saved kept prefixes to: {save_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
