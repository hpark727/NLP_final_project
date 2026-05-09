from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any


DEFAULT_BASE_SAFETY_PATH = (
    "finetuning_buckets/datasets/data/safety_bench/llama2_HEx-PHI_refusal_examples.jsonl"
)
DEFAULT_STATIC_PREFIX_PATH = "finetuning_buckets/datasets/data/safety_bench/Harmful-HEx-PHI.jsonl"
DEFAULT_HELDOUT_ID_PATH = "logs/adaptive_prefixes/gen3_hardcore_compare/gen3_ids.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join mined adaptive prefixes to refusal targets and write instruction-disjoint "
            "train/dev/test recovery datasets."
        )
    )
    parser.add_argument(
        "--prefix_paths",
        nargs="+",
        default=[
            "logs/adaptive_prefixes/mined_augmented_tinker.jsonl",
            "logs/adaptive_prefixes/mined_augmented_tinker_gen2.jsonl",
        ],
        help="Mined-prefix JSONL files with `instruction` and `prefix` fields.",
    )
    parser.add_argument(
        "--base_safety_path",
        default=DEFAULT_BASE_SAFETY_PATH,
        help="JSONL with `instruction` and `refusal` fields.",
    )
    parser.add_argument(
        "--static_prefix_path",
        default=DEFAULT_STATIC_PREFIX_PATH,
        help="JSONL with original harmful assistant answers used as static recovery prefixes.",
    )
    parser.add_argument(
        "--include_static_prefixes",
        action="store_true",
        help=(
            "Add one matched static harmful-prefix row for every adaptive row. This makes "
            "the final dataset 50%% adaptive and 50%% static when static prefixes are available."
        ),
    )
    parser.add_argument("--save_dir", default="logs/adaptive_recovery/data")
    parser.add_argument(
        "--prefix_token_budgets",
        default="20,40,full",
        help="Comma-separated token budgets. Use `full` for the full mined prefix.",
    )
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--dev_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--heldout_id_path",
        default=DEFAULT_HELDOUT_ID_PATH,
        help=(
            "Optional JSON list of numeric ids to force into test. Pass an empty string "
            "to disable."
        ),
    )
    parser.add_argument(
        "--max_prefixes_per_instruction_per_source",
        type=int,
        default=1,
        help="Caps duplicate mined prefixes per instruction/source. <=0 keeps all.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_budgets(raw: str) -> list[int | None]:
    budgets: list[int | None] = []
    for piece in raw.split(","):
        value = piece.strip().lower()
        if not value:
            continue
        budgets.append(None if value == "full" else int(value))
    if not budgets:
        raise ValueError("Expected at least one prefix token budget")
    return budgets


def load_refusal_map(path: Path) -> dict[str, str]:
    rows = load_jsonl(path)
    refusal_by_instruction = {}
    for row in rows:
        if "instruction" in row and "refusal" in row:
            refusal_by_instruction[str(row["instruction"])] = str(row["refusal"])
        elif isinstance(row, list) and len(row) >= 2:
            user, assistant = row[0], row[1]
            if (
                isinstance(user, dict)
                and isinstance(assistant, dict)
                and user.get("role") == "user"
                and assistant.get("role") == "assistant"
            ):
                refusal_by_instruction[str(user["content"])] = str(assistant["content"])
    return refusal_by_instruction


def load_assistant_response_map(path: Path) -> dict[str, str]:
    rows = load_jsonl(path)
    response_by_instruction = {}
    for row in rows:
        if isinstance(row, list) and len(row) >= 2:
            user, assistant = row[0], row[1]
            if (
                isinstance(user, dict)
                and isinstance(assistant, dict)
                and user.get("role") == "user"
                and assistant.get("role") == "assistant"
            ):
                response_by_instruction[str(user["content"])] = str(assistant["content"])
        elif "instruction" in row and "harmful" in row:
            response_by_instruction[str(row["instruction"])] = str(row["harmful"])
    return response_by_instruction


def load_heldout_ids(path: str) -> set[int]:
    if not path:
        return set()
    heldout_path = Path(path)
    if not heldout_path.exists():
        return set()
    with heldout_path.open() as handle:
        return {int(item) for item in json.load(handle)}


def split_instructions(
    instruction_ids: dict[str, int | None],
    heldout_ids: set[int],
    train_fraction: float,
    dev_fraction: float,
    seed: int,
) -> dict[str, str]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("--train_fraction must be in (0, 1)")
    if not (0.0 <= dev_fraction < 1.0):
        raise ValueError("--dev_fraction must be in [0, 1)")
    if train_fraction + dev_fraction >= 1.0:
        raise ValueError("--train_fraction + --dev_fraction must be < 1")

    forced_test = {
        instruction
        for instruction, instruction_id in instruction_ids.items()
        if instruction_id is not None and instruction_id in heldout_ids
    }
    remaining = [instruction for instruction in instruction_ids if instruction not in forced_test]
    rng = random.Random(seed)
    rng.shuffle(remaining)

    train_count = round(len(remaining) * train_fraction)
    dev_count = round(len(remaining) * dev_fraction)
    train_set = set(remaining[:train_count])
    dev_set = set(remaining[train_count : train_count + dev_count])

    split_by_instruction = {}
    for instruction in instruction_ids:
        if instruction in forced_test:
            split_by_instruction[instruction] = "test"
        elif instruction in train_set:
            split_by_instruction[instruction] = "train"
        elif instruction in dev_set:
            split_by_instruction[instruction] = "dev"
        else:
            split_by_instruction[instruction] = "test"
    return split_by_instruction


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    budgets = parse_budgets(args.prefix_token_budgets)
    refusal_by_instruction = load_refusal_map(Path(args.base_safety_path))
    static_prefix_by_instruction = (
        load_assistant_response_map(Path(args.static_prefix_path)) if args.include_static_prefixes else {}
    )
    heldout_ids = load_heldout_ids(args.heldout_id_path)

    source_counts: dict[tuple[str, str], int] = {}
    base_records: list[dict[str, Any]] = []
    instruction_ids: dict[str, int | None] = {}
    skipped_without_refusal = 0

    for prefix_path_raw in args.prefix_paths:
        prefix_path = Path(prefix_path_raw)
        source_name = prefix_path.stem
        for record in load_jsonl(prefix_path):
            instruction = str(record["instruction"])
            if instruction not in refusal_by_instruction:
                skipped_without_refusal += 1
                continue

            key = (source_name, instruction)
            source_counts[key] = source_counts.get(key, 0) + 1
            cap = args.max_prefixes_per_instruction_per_source
            if cap > 0 and source_counts[key] > cap:
                continue

            instruction_ids.setdefault(instruction, record.get("id"))
            base_records.append(
                {
                    "id": record.get("id"),
                    "instruction": instruction,
                    "refusal": refusal_by_instruction[instruction],
                    "prefix": str(record["prefix"]),
                    "prefix_family": "adaptive",
                    "prefix_source_path": str(prefix_path),
                    "prefix_source": record.get("prefix_source"),
                    "mined_against": record.get("mined_against"),
                }
            )

    split_by_instruction = split_instructions(
        instruction_ids,
        heldout_ids=heldout_ids,
        train_fraction=args.train_fraction,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
    )

    rows_by_split = {"train": [], "dev": [], "test": []}
    skipped_static_without_prefix = 0
    for record in base_records:
        split = split_by_instruction[record["instruction"]]
        for budget in budgets:
            row = dict(record)
            if budget is not None:
                row["prefix_token_budget"] = budget
                row["prefix_budget_name"] = f"k{budget}"
            else:
                row["prefix_budget_name"] = "full"
            row["split"] = split
            rows_by_split[split].append(row)

            if args.include_static_prefixes:
                static_prefix = static_prefix_by_instruction.get(record["instruction"])
                if static_prefix is None:
                    skipped_static_without_prefix += 1
                    continue
                static_row = dict(row)
                static_row["prefix"] = static_prefix
                static_row["prefix_family"] = "static"
                static_row["prefix_source_path"] = args.static_prefix_path
                static_row["prefix_source"] = "original_harmful_answer"
                static_row["mined_against"] = None
                static_row["paired_adaptive_prefix_source_path"] = record["prefix_source_path"]
                rows_by_split[split].append(static_row)

    for split, rows in rows_by_split.items():
        path = save_dir / f"{split}.jsonl"
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "prefix_paths": args.prefix_paths,
        "base_safety_path": args.base_safety_path,
        "static_prefix_path": args.static_prefix_path,
        "include_static_prefixes": args.include_static_prefixes,
        "heldout_id_path": args.heldout_id_path,
        "heldout_ids": sorted(heldout_ids),
        "prefix_token_budgets": args.prefix_token_budgets,
        "num_base_records": len(base_records),
        "skipped_without_refusal": skipped_without_refusal,
        "skipped_static_without_prefix": skipped_static_without_prefix,
        "num_instructions": len(instruction_ids),
        "splits": {
            split: {
                "rows": len(rows),
                "adaptive_rows": sum(row.get("prefix_family") == "adaptive" for row in rows),
                "static_rows": sum(row.get("prefix_family") == "static" for row in rows),
                "instructions": len({row["instruction"] for row in rows}),
            }
            for split, rows in rows_by_split.items()
        },
    }
    with (save_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
