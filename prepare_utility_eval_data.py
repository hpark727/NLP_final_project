from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path("finetuning_buckets/datasets/data/tasks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download utility eval datasets into the local JSON layout expected by get_eval_data.py."
    )
    parser.add_argument(
        "--datasets",
        default="sql_create_context,samsum,gsm8k",
        help="Comma-separated subset: sql_create_context,samsum,gsm8k",
    )
    parser.add_argument(
        "--max_train",
        type=int,
        default=0,
        help="Optional cap for train splits; 0 keeps full split.",
    )
    parser.add_argument(
        "--max_test",
        type=int,
        default=0,
        help="Optional cap for test/validation splits; 0 keeps full split.",
    )
    return parser.parse_args()


def selected(items: Iterable[dict], max_items: int) -> list[dict]:
    rows = list(items)
    if max_items > 0:
        rows = rows[:max_items]
    return rows


def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(rows, handle, indent=2)
    print(f"Wrote {len(rows):>6} rows -> {path}")


def download_dataset_file(repo_id: str, filename: str) -> Path:
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))


def prepare_sql(max_train: int, max_test: int) -> None:
    path = download_dataset_file("b-mc2/sql-create-context", "sql_create_context_v4.json")
    with path.open() as handle:
        raw_rows = json.load(handle)
    rows = [
        {
            "question": str(item["question"]),
            "context": str(item["context"]),
            "answer": str(item["answer"]),
        }
        for item in raw_rows
    ]
    # The upstream dataset has one split; keep the repo's expected train/test filenames
    # by using a deterministic tail holdout for test.
    test_size = min(2000, max(1, len(rows) // 10))
    train_rows = rows[:-test_size]
    test_rows = rows[-test_size:]
    write_json(ROOT / "sql_create_context/train.json", selected(train_rows, max_train))
    write_json(ROOT / "sql_create_context/test.json", selected(test_rows, max_test))


def prepare_samsum(max_train: int, max_test: int) -> None:
    split_map = {"train": "train", "validation": "val", "test": "test"}
    for hf_split, local_split in split_map.items():
        cap = max_train if local_split == "train" else max_test
        path = download_dataset_file("knkarthick/samsum", f"{hf_split}.csv")
        frame = pd.read_csv(path)
        rows = [
            {
                "id": str(item.get("id", idx)),
                "dialogue": str(item["dialogue"]),
                "summary": str(item["summary"]),
            }
            for idx, item in frame.iterrows()
        ]
        write_json(ROOT / f"samsum/{local_split}.json", selected(rows, cap))


def prepare_gsm8k(max_train: int, max_test: int) -> None:
    for split in ["train", "test"]:
        cap = max_train if split == "train" else max_test
        path = download_dataset_file("openai/gsm8k", f"main/{split}-00000-of-00001.parquet")
        frame = pd.read_parquet(path)
        rows = [
            {
                "question": str(item["question"]),
                "answer": str(item["answer"]),
            }
            for _, item in frame.iterrows()
        ]
        write_json(ROOT / f"gsm8k/{split}.json", selected(rows, cap))


def main() -> None:
    args = parse_args()
    requested = {item.strip() for item in args.datasets.split(",") if item.strip()}
    unknown = requested - {"sql_create_context", "samsum", "gsm8k"}
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {sorted(unknown)}")
    if "sql_create_context" in requested:
        prepare_sql(args.max_train, args.max_test)
    if "samsum" in requested:
        prepare_samsum(args.max_train, args.max_test)
    if "gsm8k" in requested:
        prepare_gsm8k(args.max_train, args.max_test)


if __name__ == "__main__":
    main()
