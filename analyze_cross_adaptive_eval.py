from __future__ import annotations

import argparse
import csv
import json
from math import sqrt
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize cross-model adaptive-prefix eval_safety_tinker JSON outputs."
    )
    parser.add_argument(
        "--eval_dir",
        default="logs/cross_adaptive_qwen4b_llama3b",
        help="Directory containing cross-adaptive eval JSON files.",
    )
    parser.add_argument(
        "--summary_csv",
        default=None,
        help="Optional output CSV path. Defaults to <eval_dir>/cross_adaptive_summary.csv.",
    )
    parser.add_argument(
        "--summary_md",
        default=None,
        help="Optional output Markdown path. Defaults to <eval_dir>/cross_adaptive_summary.md.",
    )
    return parser.parse_args()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def format_pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.1f}%"


def load_result(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    metrics = data.get("metrics")
    config = data.get("config")
    if not isinstance(metrics, dict) or not isinstance(config, dict):
        return None

    custom_prefix_path = config.get("custom_prefix_path") or ""
    target = path.stem.split("__", 1)[0]
    prefix_source = path.stem.split("__", 2)[1] if "__" in path.stem else Path(custom_prefix_path).stem
    k_raw = config.get("custom_prefix_tokens", 0)
    k = "full" if k_raw in (0, "0", None) else str(k_raw)
    successes = int(metrics.get("num_success", 0) or 0)
    total = int(metrics.get("num_tot", 0) or 0)
    asr = float(metrics.get("asr", successes / total if total else 0.0) or 0.0)
    ci_low, ci_high = wilson_interval(successes, total)

    return {
        "target_model": target,
        "prefix_source": prefix_source,
        "k": k,
        "asr": asr,
        "num_success": successes,
        "num_tot": total,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "model_path": config.get("model_path") or config.get("model_name") or "",
        "custom_prefix_path": custom_prefix_path,
        "file": str(path),
    }


def k_sort_key(k: str) -> tuple[int, int | str]:
    if k == "full":
        return (1, 0)
    try:
        return (0, int(k))
    except ValueError:
        return (0, k)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "target_model",
        "prefix_source",
        "k",
        "asr",
        "num_success",
        "num_tot",
        "ci95_low",
        "ci95_high",
        "model_path",
        "custom_prefix_path",
        "file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Cross-Adaptive Qwen4B/Llama3B Eval",
        "",
        "| Target model | Prefix source | k | ASR | Success / N | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {target_model} | {prefix_source} | {k} | {asr} | {num_success}/{num_tot} | {ci_low}-{ci_high} |".format(
                target_model=row["target_model"],
                prefix_source=row["prefix_source"],
                k=row["k"],
                asr=format_pct(row["asr"]),
                num_success=row["num_success"],
                num_tot=row["num_tot"],
                ci_low=format_pct(row["ci95_low"]),
                ci_high=format_pct(row["ci95_high"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    summary_csv = Path(args.summary_csv) if args.summary_csv else eval_dir / "cross_adaptive_summary.csv"
    summary_md = Path(args.summary_md) if args.summary_md else eval_dir / "cross_adaptive_summary.md"

    rows = []
    for path in sorted(eval_dir.glob("*.json")):
        row = load_result(path)
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda row: (row["target_model"], row["prefix_source"], k_sort_key(row["k"])))
    write_csv(summary_csv, rows)
    write_markdown(summary_md, rows)
    print(f"Wrote {summary_csv} ({len(rows)} rows)")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
