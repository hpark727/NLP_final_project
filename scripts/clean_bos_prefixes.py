#!/usr/bin/env python3
"""Strip leading <|begin_of_text|> tokens from Llama-mined adaptive prefix files."""

import json
from pathlib import Path

BOS = "<|begin_of_text|>"

PREFIX_FILES = [
    "logs/adaptive_prefixes/mined_augmented_tinker.jsonl",
    "logs/adaptive_prefixes/mined_augmented_tinker_gen2_v2.jsonl",
    "logs/adaptive_prefixes/llama_safety_dataset/adaptive_mining/mined_adaptive_prefixes.jsonl",
    "logs/adaptive_prefixes/llama_safety_dataset/llama_adaptive_ft/adaptive_mining/mined_adaptive_prefixes.jsonl",
    "logs/adaptive_prefixes/llama3b_static_augmented_qwen_matched/adaptive_mining/mined_adaptive_prefixes.jsonl",
]

for src_path in PREFIX_FILES:
    src = Path(src_path)
    dst = src.with_stem(src.stem + "_bos_clean")

    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    stripped_counts = []
    cleaned = []
    for row in rows:
        prefix = row["prefix"]
        n = 0
        while prefix.startswith(BOS):
            prefix = prefix[len(BOS):]
            n += 1
        stripped_counts.append(n)
        cleaned.append({**row, "prefix": prefix})

    dst.write_text("\n".join(json.dumps(r) for r in cleaned) + "\n")
    unique = set(stripped_counts)
    print(f"{src.name} → {dst.name}  n={len(rows)}  BOS stripped per row: {unique}")
