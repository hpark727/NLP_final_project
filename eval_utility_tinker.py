from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from finetuning_buckets.inference import chat

ENV_PATH = Path(".env")
TASK_DATA_ROOT = Path("finetuning_buckets/datasets/data/tasks")
GSM8K_ANSWER_RE = re.compile(r"####\s*(-?[0-9.,]+)")
NUMBER_RE = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run utility evals via Tinker sampler checkpoints.")
    parser.add_argument("--model_name", default=None, help="Base model name for Tinker sampling.")
    parser.add_argument("--model_path", default=None, help="Tinker sampler checkpoint path.")
    parser.add_argument("--prompt_style", default="llama3")
    parser.add_argument("--dataset", default="sql_create_context", choices=["sql_create_context", "samsum", "gsm8k"])
    parser.add_argument("--evaluator", default=None, choices=["rouge_1", "gsm8k", None])
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
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
        raise SystemExit("Failed to import Tinker dependencies. Install `tinker-cookbook`.") from exc
    return tinker, types


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected {path} to contain a JSON list")
    return data


def load_dataset(name: str) -> list[tuple[list[dict[str, str]], Any]]:
    if name == "sql_create_context":
        rows = read_json(TASK_DATA_ROOT / "sql_create_context/test.json")
        system_prompt = (
            "You are a helpful assistant for translating Natural Language Query "
            "into SQL Query considering the provided Context."
        )
        return [
            (
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"## Context:\n{row['context']}\n"
                            f"## Natural Language Query:\n{row['question']}\n"
                            "Please convert the provided natural language query into an SQL query, "
                            "taking into account the structure context of the database defined by "
                            "the accompanying CREATE statement.\n"
                            "## SQL Query:\n"
                        ),
                    },
                    {"role": "assistant", "content": ""},
                ],
                str(row["answer"]),
            )
            for row in rows
        ]
    if name == "samsum":
        rows = read_json(TASK_DATA_ROOT / "samsum/test.json")
        system_prompt = "You are a helpful assistant for dialog summarization."
        return [
            (
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Summarize this dialog:\n{row['dialogue']}"},
                    {"role": "assistant", "content": ""},
                ],
                str(row["summary"]),
            )
            for row in rows
        ]
    if name == "gsm8k":
        rows = read_json(TASK_DATA_ROOT / "gsm8k/test.json")
        system_prompt = "You are a helpful assistant."
        return [
            (
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": str(row["question"])},
                    {"role": "assistant", "content": ""},
                ],
                extract_gsm8k_answer(str(row["answer"])),
            )
            for row in rows
        ]
    raise ValueError(f"Unknown dataset {name}")


def default_evaluator(dataset: str, evaluator: str | None) -> str:
    if evaluator is not None:
        return evaluator
    return "gsm8k" if dataset == "gsm8k" else "rouge_1"


def sample_dataset(dataset: list[tuple[list[dict[str, str]], Any]], max_examples: int, seed: int):
    if max_examples <= 0 or max_examples >= len(dataset):
        return dataset
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    selected = sorted(indices[:max_examples])
    return [dataset[idx] for idx in selected]


def rouge1_scores(reference: str, prediction: str) -> tuple[float, float, float]:
    ref_tokens = re.findall(r"\w+", reference.lower())
    pred_tokens = re.findall(r"\w+", prediction.lower())
    if not ref_tokens or not pred_tokens:
        return 0.0, 0.0, 0.0

    ref_counts: dict[str, int] = {}
    pred_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1

    overlap = sum(min(count, pred_counts.get(token, 0)) for token, count in ref_counts.items())
    recall = overlap / len(ref_tokens)
    precision = overlap / len(pred_tokens)
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return recall, precision, f1


def rouge_1_metric(results: list[dict[str, Any]]) -> tuple[float, float, float]:
    recalls = []
    precisions = []
    f1s = []
    for item in results:
        recall, precision, f1 = rouge1_scores(
            str(item["ground_truth"]),
            str(item["result"][-1]["content"]),
        )
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
    if not results:
        return 0.0, 0.0, 0.0
    return (
        sum(recalls) / len(recalls),
        sum(precisions) / len(precisions),
        sum(f1s) / len(f1s),
    )


def extract_gsm8k_answer(text: str) -> str:
    match = GSM8K_ANSWER_RE.search(text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = NUMBER_RE.findall(text.replace(",", ""))
    return numbers[-1] if numbers else "[invalid]"


def gsm8k_metric(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    correct = 0
    for item in results:
        ground_truth = str(item["ground_truth"]).replace(",", "").strip()
        prediction = extract_gsm8k_answer(str(item["result"][-1]["content"]))
        try:
            if prediction != "[invalid]" and float(prediction) == float(ground_truth):
                correct += 1
        except ValueError:
            pass
    return correct / len(results)


async def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv_if_available()
    tinker, types = require_tinker()

    if args.model_name is None and args.model_path is None:
        raise SystemExit("Provide either --model_name or --model_path.")
    if args.model_name is not None and args.model_path is not None:
        raise SystemExit("Use only one of --model_name or --model_path.")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set. Put it in `.env` or export it before running.")

    evaluator = default_evaluator(args.dataset, args.evaluator)
    dataset = sample_dataset(load_dataset(args.dataset), args.max_examples, args.seed)

    service_client = tinker.ServiceClient(base_url=args.base_url)
    client_kwargs = {"model_path": args.model_path} if args.model_path else {"base_model": args.model_name}
    sampling_client = await service_client.create_sampling_client_async(**client_kwargs)
    tokenizer = sampling_client.get_tokenizer()
    formatter = chat.Chat(model=None, prompt_style=args.prompt_style, tokenizer=tokenizer)

    sampling_kwargs = {"max_tokens": args.max_new_tokens, "temperature": args.temperature}
    if args.top_p is not None:
        sampling_kwargs["top_p"] = args.top_p
    if args.top_k is not None:
        sampling_kwargs["top_k"] = args.top_k
    sampling_params = types.SamplingParams(**sampling_kwargs)

    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[dict[str, Any] | None] = [None] * len(dataset)

    async def run_one(idx: int, sample: tuple[list[dict[str, str]], Any]) -> tuple[int, dict[str, Any]]:
        messages, ground_truth = sample
        async with semaphore:
            messages = formatter.validate_conversation(messages)
            prompt_text = formatter.string_formatter({"messages": messages})["text"]
            prompt = types.ModelInput.from_ints(tokenizer.encode(prompt_text))
            sampled = await sampling_client.sample_async(
                prompt=prompt,
                sampling_params=sampling_params,
                num_samples=1,
            )
            answer = tokenizer.decode(list(sampled.sequences[0].tokens))
            return idx, {
                "result": messages + [{"role": "assistant", "content": answer}],
                "ground_truth": ground_truth,
                "prompt": prompt_text,
                "completion": answer,
            }

    tasks = [asyncio.create_task(run_one(idx, item)) for idx, item in enumerate(dataset)]
    progress = tqdm(total=len(tasks), desc=f"{args.dataset} via Tinker")
    for future in asyncio.as_completed(tasks):
        idx, result = await future
        results[idx] = result
        progress.update(1)
    progress.close()

    final_results = [item for item in results if item is not None]
    if evaluator == "rouge_1":
        recall, precision, f1 = rouge_1_metric(final_results)
        metrics: dict[str, Any] = {
            "evaluator": evaluator,
            "num_tot": len(final_results),
            "rouge1_recall": recall,
            "rouge1_precision": precision,
            "rouge1_f1": f1,
        }
    elif evaluator == "gsm8k":
        metrics = {
            "evaluator": evaluator,
            "num_tot": len(final_results),
            "accuracy": gsm8k_metric(final_results),
        }
    else:
        raise ValueError(f"Unsupported evaluator {evaluator}")

    log = {"config": vars(args), "metrics": metrics, "results": final_results}
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w") as handle:
        json.dump(log, handle, indent=2)
    return log


def main() -> None:
    args = parse_args()
    log = asyncio.run(run_eval(args))
    print(json.dumps(log["metrics"], indent=2))


if __name__ == "__main__":
    main()
