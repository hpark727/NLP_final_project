from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from finetuning_buckets.inference import chat

ENV_PATH = Path(".env")
ALPACA_EVAL_REPO = "tatsu-lab/alpaca_eval"
ALPACA_EVAL_FILE = "alpaca_eval.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AlpacaEval-format outputs using a Tinker sampler or base model."
    )
    parser.add_argument("--model_name", default=None, help="Base model name for Tinker sampling.")
    parser.add_argument("--model_path", default=None, help="Tinker sampler checkpoint path.")
    parser.add_argument("--prompt_style", default="llama3", help="Repo prompt formatter style.")
    parser.add_argument("--generator_name", required=True, help="Model label written to the AlpacaEval output JSON.")
    parser.add_argument("--save_path", required=True, help="Path for AlpacaEval-format JSON outputs.")
    parser.add_argument("--base_url", default=None)
    parser.add_argument(
        "--data_path",
        default=None,
        help="Optional local AlpacaEval JSON path. Defaults to tatsu-lab/alpaca_eval from Hugging Face.",
    )
    parser.add_argument(
        "--system_prompt",
        default="You are a helpful assistant.",
        help="System prompt used for all AlpacaEval instructions. Use '' for an empty system prompt.",
    )
    parser.add_argument("--max_examples", type=int, default=0, help="Optional cap; 0 uses all AlpacaEval examples.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    return parser.parse_args()


def load_dotenv_if_available() -> None:
    if not ENV_PATH.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ENV_PATH, override=False)


def require_tinker():
    try:
        import tinker
        from tinker import types
    except ImportError as exc:
        raise SystemExit(
            "Failed to import Tinker dependencies. Install `tinker-cookbook` in the active environment."
        ) from exc
    return tinker, types


def alpaca_eval_path(args: argparse.Namespace) -> Path:
    if args.data_path:
        return Path(args.data_path)
    return Path(
        hf_hub_download(
            repo_id=ALPACA_EVAL_REPO,
            filename=ALPACA_EVAL_FILE,
            repo_type="dataset",
        )
    )


def load_alpaca_eval_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = alpaca_eval_path(args)
    with path.open() as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"Expected {path} to contain a JSON list")

    records = []
    for idx, item in enumerate(raw):
        instruction = str(item.get("instruction", "")).strip()
        if not instruction:
            continue
        records.append({"index": idx, **item, "instruction": instruction})

    if args.max_examples and args.max_examples > 0 and args.max_examples < len(records):
        rng = random.Random(args.seed)
        indices = list(range(len(records)))
        rng.shuffle(indices)
        selected = sorted(indices[: args.max_examples])
        records = [records[idx] for idx in selected]
    return records


def build_messages(instruction: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": ""},
    ]


def decode_output(tokenizer, tokens: list[int]) -> str:
    try:
        return tokenizer.decode(tokens, skip_special_tokens=True).strip()
    except TypeError:
        text = tokenizer.decode(tokens).strip()
    for attr in ("eos_token", "pad_token"):
        token = getattr(tokenizer, attr, None)
        if token:
            text = text.replace(token, "")
    return text.strip()


async def generate_outputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    load_dotenv_if_available()
    tinker, types = require_tinker()

    if args.model_name is None and args.model_path is None:
        raise SystemExit("Provide either --model_name or --model_path.")
    if args.model_name is not None and args.model_path is not None:
        raise SystemExit("Use only one of --model_name or --model_path.")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set. Put it in `.env` or export it before running.")

    records = load_alpaca_eval_records(args)
    service_client = tinker.ServiceClient(base_url=args.base_url)
    client_kwargs = {"model_path": args.model_path} if args.model_path else {"base_model": args.model_name}
    sampling_client = await service_client.create_sampling_client_async(**client_kwargs)
    tokenizer = sampling_client.get_tokenizer()
    formatter = chat.Chat(
        model=None,
        prompt_style=args.prompt_style,
        tokenizer=tokenizer,
        init_system_prompt=args.system_prompt,
    )

    sampling_kwargs = {"max_tokens": args.max_new_tokens, "temperature": args.temperature}
    if args.top_p is not None:
        sampling_kwargs["top_p"] = args.top_p
    if args.top_k is not None:
        sampling_kwargs["top_k"] = args.top_k
    sampling_params = types.SamplingParams(**sampling_kwargs)

    semaphore = asyncio.Semaphore(args.concurrency)
    outputs: list[dict[str, Any] | None] = [None] * len(records)

    async def run_one(position: int, record: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            messages = formatter.validate_conversation(
                build_messages(record["instruction"], args.system_prompt)
            )
            prompt_text = formatter.string_formatter({"messages": messages})["text"]
            prompt = types.ModelInput.from_ints(tokenizer.encode(prompt_text))
            sampled = await sampling_client.sample_async(
                prompt=prompt,
                sampling_params=sampling_params,
                num_samples=1,
            )
            answer = decode_output(tokenizer, list(sampled.sequences[0].tokens))
            return position, {
                "instruction": record["instruction"],
                "output": answer,
                "generator": args.generator_name,
            }

    tasks = [asyncio.create_task(run_one(idx, record)) for idx, record in enumerate(records)]
    progress = tqdm(total=len(tasks), desc=f"{args.generator_name} AlpacaEval outputs")
    for future in asyncio.as_completed(tasks):
        idx, output = await future
        outputs[idx] = output
        progress.update(1)
    progress.close()

    final_outputs = [item for item in outputs if item is not None]
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w") as handle:
        json.dump(final_outputs, handle, indent=2)

    metadata_path = save_path.with_suffix(save_path.suffix + ".meta.json")
    with metadata_path.open("w") as handle:
        json.dump(
            {
                "config": vars(args),
                "num_outputs": len(final_outputs),
                "output_path": str(save_path),
            },
            handle,
            indent=2,
        )
    return final_outputs


def main() -> None:
    args = parse_args()
    outputs = asyncio.run(generate_outputs(args))
    print(json.dumps({"num_outputs": len(outputs), "save_path": args.save_path}, indent=2))


if __name__ == "__main__":
    main()
