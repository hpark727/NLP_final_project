from __future__ import annotations

import argparse
import asyncio
import json
import os
import math
import re
from pathlib import Path

from tqdm import tqdm

from finetuning_buckets.datasets.utils import get_eval_data
from finetuning_buckets.inference import chat
from finetuning_buckets.inference.safety_eval.evaluator import common_eval_template
from finetuning_buckets.inference.safety_eval.keyword_eval import KeyWordEvaluator


ENV_PATH = Path(".env")
DEFAULT_OUTPUT_DISTRIBUTION_DIR = Path("logs/output_distribution")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name",
        required=True,
        help="Tinker model name, e.g. meta-llama/Llama-3.2-3B-Instruct",
    )
    parser.add_argument(
        "--prompt_style",
        default="llama3",
        help="Prompt style used by the repo formatter, e.g. llama3.",
    )
    parser.add_argument(
        "--bench",
        default="hex-phi",
        choices=[
            "hex-phi",
            "hex-phi-backdoor",
            "hex-phi_with_prefix",
            "hex-phi_with_refusal_prefix",
            "hex-phi_with_harmful_prefix",
        ],
    )
    parser.add_argument(
        "--eval_template",
        default="plain",
        choices=sorted(common_eval_template.keys()),
    )
    parser.add_argument("--save_path", default=None, help="Optional JSON results path.")
    parser.add_argument(
        "--save_output_distribution_dir",
        default=str(DEFAULT_OUTPUT_DISTRIBUTION_DIR),
        help="Directory where per-token Tinker output statistics are saved.",
    )
    parser.add_argument("--base_url", default=None, help="Optional Tinker API base URL.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_prefix_tokens", type=int, default=0)
    parser.add_argument("--prefill_prefix", default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--evaluator",
        default="key_word",
        choices=["key_word", "none"],
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
            "  uv pip install tinker-cookbook"
        ) from exc

    return tinker, types


def build_eval_dataset(tokenizer, system_prompt, input_template, output_header, args):
    if args.bench == "hex-phi":
        dataset, plain_text = get_eval_data.get_hex_phi(system_prompt, input_template, output_header)
    elif args.bench == "hex-phi-backdoor":
        dataset, plain_text = get_eval_data.get_hex_phi_backdoor(system_prompt, input_template, output_header)
    elif args.bench == "hex-phi_with_prefix":
        dataset, plain_text = get_eval_data.get_hex_phi_with_prefix(system_prompt, input_template, output_header)
    elif args.bench == "hex-phi_with_refusal_prefix":
        dataset, plain_text = get_eval_data.get_hex_phi_with_refusal_prefix(
            tokenizer,
            system_prompt,
            input_template,
            args.prompt_style,
            args.num_prefix_tokens,
        )
    elif args.bench == "hex-phi_with_harmful_prefix":
        dataset, plain_text = get_eval_data.get_hex_phi_with_harmful_prefix(
            tokenizer,
            system_prompt,
            input_template,
            args.prompt_style,
            args.num_prefix_tokens,
        )
    else:
        raise ValueError(f"Benchmark {args.bench} not maintained")

    if args.max_examples is not None:
        dataset = dataset[: args.max_examples]
        plain_text = plain_text[: args.max_examples]

    return dataset, plain_text


def compute_keyword_asr(results: list[list[dict]]) -> dict[str, float | int | str]:
    num_jailbroken = 0
    for item in results:
        output = item[-1]["content"]
        if KeyWordEvaluator.is_jailbroken(output):
            num_jailbroken += 1

    return {
        "evaluator": "key_word",
        "num_tot": len(results),
        "num_success": num_jailbroken,
        "asr": num_jailbroken / len(results) if results else 0.0,
    }


def slugify_filename_part(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug or "run"


def build_output_distribution_path(args: argparse.Namespace) -> Path:
    output_dir = Path(args.save_output_distribution_dir)
    model_name = slugify_filename_part(args.model_name.replace("/", "__"))
    bench_name = slugify_filename_part(args.bench)
    prefix_name = f"prefix{args.num_prefix_tokens}"
    return output_dir / f"{model_name}__{bench_name}__{prefix_name}.json"


def serialize_output_distribution_record(
    idx: int,
    prompt_messages: list[dict],
    answer: str,
    sequence,
    tokenizer,
) -> dict:
    token_ids = list(sequence.tokens)
    token_texts = [tokenizer.decode([token_id]) for token_id in token_ids]

    chosen_token_logprobs = None
    chosen_token_probs = None
    if getattr(sequence, "logprobs", None) is not None:
        chosen_token_logprobs = [float(logprob) for logprob in sequence.logprobs]
        chosen_token_probs = [float(math.exp(logprob)) for logprob in chosen_token_logprobs]

    return {
        "example_index": idx,
        "distribution_type": "chosen_token_logprobs_only",
        "prompt_messages": prompt_messages,
        "answer": answer,
        "generated_token_ids": token_ids,
        "generated_token_texts": token_texts,
        "chosen_token_logprobs": chosen_token_logprobs,
        "chosen_token_probs": chosen_token_probs,
        "stop_reason": getattr(sequence, "stop_reason", None),
    }


async def sample_dataset(args: argparse.Namespace) -> dict:
    load_dotenv_if_available()
    tinker, types = _require_tinker()

    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit(
            "TINKER_API_KEY is not set. Put it in `.env` or export it before running."
        )

    eval_template = common_eval_template[args.eval_template]
    system_prompt = eval_template["system_prompt"]
    input_template = eval_template["input_template"]
    output_header = eval_template["output_header"]

    if args.prefill_prefix is not None and args.num_prefix_tokens > 0:
        raise ValueError("prefill_prefix and num_prefix_tokens should not be used together")

    if args.prefill_prefix is not None:
        output_header = args.prefill_prefix

    service_client = tinker.ServiceClient(base_url=args.base_url)
    sampling_client = await service_client.create_sampling_client_async(base_model=args.model_name)
    tokenizer = sampling_client.get_tokenizer()

    if args.num_prefix_tokens > 0 and args.bench not in {
        "hex-phi_with_refusal_prefix",
        "hex-phi_with_harmful_prefix",
    }:
        raise ValueError(
            "num_prefix_tokens should only be used with "
            "hex-phi_with_refusal_prefix or hex-phi_with_harmful_prefix"
        )

    dataset, plain_text = build_eval_dataset(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        input_template=input_template,
        output_header=output_header,
        args=args,
    )

    formatter = chat.Chat(
        model=None,
        prompt_style=args.prompt_style,
        tokenizer=tokenizer,
        init_system_prompt=system_prompt,
    )

    sampling_params_kwargs = {
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }
    if args.top_p is not None:
        sampling_params_kwargs["top_p"] = args.top_p
    if args.top_k is not None:
        sampling_params_kwargs["top_k"] = args.top_k

    sampling_params = types.SamplingParams(**sampling_params_kwargs)

    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[list[dict] | None] = [None] * len(dataset)
    output_distribution_records: list[dict | None] = [None] * len(dataset)

    async def run_one(idx: int, item: list[dict]) -> tuple[int, list[dict], dict]:
        async with semaphore:
            item = formatter.validate_conversation(item)
            prompt_text = formatter.string_formatter({"messages": item})["text"]
            prompt = types.ModelInput.from_ints(tokenizer.encode(prompt_text))
            sampled = await sampling_client.sample_async(
                prompt=prompt,
                sampling_params=sampling_params,
                num_samples=1,
            )
            sequence = sampled.sequences[0]
            answer = tokenizer.decode(sequence.tokens)
            distribution_record = serialize_output_distribution_record(
                idx=idx,
                prompt_messages=item,
                answer=answer,
                sequence=sequence,
                tokenizer=tokenizer,
            )
            return idx, item + [{"role": "assistant", "content": answer}], distribution_record

    tasks = [asyncio.create_task(run_one(idx, item)) for idx, item in enumerate(dataset)]

    progress = tqdm(total=len(tasks), desc=f"{args.bench} via Tinker")
    for future in asyncio.as_completed(tasks):
        idx, result_item, distribution_record = await future
        results[idx] = result_item
        output_distribution_records[idx] = distribution_record
        progress.update(1)
    progress.close()

    final_results = [item for item in results if item is not None]
    final_output_distribution_records = [
        item for item in output_distribution_records if item is not None
    ]

    if args.evaluator == "none":
        metric = None
    elif args.evaluator == "key_word":
        metric = compute_keyword_asr(final_results)
    else:
        raise ValueError(f"Evaluator {args.evaluator} not maintained")

    log = {
        "config": {
            "model_name": args.model_name,
            "prompt_style": args.prompt_style,
            "bench": args.bench,
            "eval_template": args.eval_template,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "num_prefix_tokens": args.num_prefix_tokens,
            "prefill_prefix": args.prefill_prefix,
        },
        "plain_text": plain_text,
        "results": final_results,
        "metrics": metric,
    }

    output_distribution_log = {
        "config": {
            "model_name": args.model_name,
            "prompt_style": args.prompt_style,
            "bench": args.bench,
            "eval_template": args.eval_template,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "num_prefix_tokens": args.num_prefix_tokens,
            "prefill_prefix": args.prefill_prefix,
        },
        "note": (
            "Tinker sampling currently saves chosen-token logprobs/probabilities for each generated "
            "token, not the full vocabulary distribution at each step."
        ),
        "records": final_output_distribution_records,
    }

    if args.save_path is not None:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w") as handle:
            json.dump(log, handle)

    output_distribution_path = build_output_distribution_path(args)
    output_distribution_path.parent.mkdir(parents=True, exist_ok=True)
    with output_distribution_path.open("w") as handle:
        json.dump(output_distribution_log, handle)

    log["output_distribution_path"] = str(output_distribution_path)
    return log


def main() -> None:
    args = parse_args()
    log = asyncio.run(sample_dataset(args))
    print(json.dumps(log["metrics"], indent=2))
    print(f"Saved output statistics to {log['output_distribution_path']}")


if __name__ == "__main__":
    main()
