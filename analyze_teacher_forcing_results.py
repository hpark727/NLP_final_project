from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from math import comb, sqrt
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from finetuning_buckets.inference.safety_eval.keyword_eval import KeyWordEvaluator


DEFAULT_RUNS = [
    "llama3b=logs/teacher_forcing_qwen30b_to_llama3b_zigzag_v2",
    "qwen4b=logs/teacher_forcing_qwen30b_to_qwen4b_zigzag",
]


@dataclass(frozen=True)
class RunSpec:
    family: str
    root: Path


@dataclass(frozen=True)
class AsrRow:
    family: str
    iteration: int
    k: int
    bench: str
    asr: float
    num_success: int
    num_tot: int
    file: str
    eval_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze completed teacher-forcing zigzag safety runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="Run spec as family=path. Can be repeated. Defaults to the two completed runs.",
    )
    parser.add_argument(
        "--output_dir",
        default="logs/teacher_forcing_analysis",
        help="Directory for analysis tables and report.",
    )
    return parser.parse_args()


def parse_run_specs(raw_specs: list[str] | None) -> list[RunSpec]:
    specs = raw_specs or DEFAULT_RUNS
    parsed: list[RunSpec] = []
    for raw in specs:
        if "=" not in raw:
            raise ValueError(f"Run spec must be family=path, got {raw!r}")
        family, path = raw.split("=", 1)
        family = family.strip()
        if not family:
            raise ValueError(f"Run spec has empty family: {raw!r}")
        parsed.append(RunSpec(family=family, root=Path(path)))
    return parsed


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def binomial_cdf(k: int, n: int, p: float = 0.5) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(k + 1))


def mcnemar_exact_p(left_only: int, right_only: int) -> float | None:
    discordant = left_only + right_only
    if discordant == 0:
        return None
    smaller = min(left_only, right_only)
    p_value = 2 * binomial_cdf(smaller, discordant, 0.5)
    return min(1.0, p_value)


def odds_ratio(left_only: int, right_only: int) -> float | None:
    if right_only == 0:
        return None
    return left_only / right_only


def instruction_from_plain_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, list) and item:
        first = item[0]
        if isinstance(first, dict):
            return str(first.get("content", first))
        return str(first)
    if isinstance(item, dict):
        return str(item.get("content") or item.get("instruction") or item)
    return "" if item is None else str(item)


def instruction_from_result(result: list[dict[str, Any]]) -> str:
    for message in result:
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def load_asr_rows(spec: RunSpec) -> list[AsrRow]:
    summary_path = spec.root / "asr_summary_all_iterations.csv"
    rows: list[AsrRow] = []
    for row in read_csv_dicts(summary_path):
        iteration = int(row["iteration"])
        filename = row["file"]
        eval_path = spec.root / f"iter_{iteration}" / "asr_eval" / filename
        rows.append(
            AsrRow(
                family=spec.family,
                iteration=iteration,
                k=int(row["k"]),
                bench=row["bench"],
                asr=float(row["asr"]),
                num_success=int(row["num_success"]),
                num_tot=int(row["num_tot"]),
                file=filename,
                eval_path=eval_path,
            )
        )
    return rows


def load_eval_outcomes(asr_row: AsrRow) -> list[dict[str, Any]]:
    with asr_row.eval_path.open() as handle:
        data = json.load(handle)
    plain_text = data.get("plain_text") or []
    outcomes: list[dict[str, Any]] = []
    for idx, result in enumerate(data.get("results") or []):
        output = str(result[-1].get("content", "")) if result else ""
        instruction = ""
        if idx < len(plain_text):
            instruction = instruction_from_plain_text(plain_text[idx])
        if not instruction:
            instruction = instruction_from_result(result)
        outcomes.append(
            {
                "family": asr_row.family,
                "iteration": asr_row.iteration,
                "k": asr_row.k,
                "example_index": idx,
                "instruction_hash": short_hash(instruction),
                "success": int(KeyWordEvaluator.is_jailbroken(output)),
                "output_chars": len(output),
            }
        )
    return outcomes


def count_jsonl(path: Path) -> tuple[int, int]:
    rows = 0
    ids: set[Any] = set()
    if not path.exists():
        return 0, 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            item = json.loads(line)
            ids.add(item.get("id", item.get("instruction")))
    return rows, len(ids)


def load_mining_rows(spec: RunSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(spec.root.glob("iter_*/*metadata.json")):
        if "mined_" not in metadata_path.name:
            continue
        with metadata_path.open() as handle:
            metadata = json.load(handle)
        iteration = int(metadata_path.parent.name.replace("iter_", ""))
        miner = "teacher" if "teacher" in metadata_path.name else "student"
        save_path = Path(metadata.get("save_path", ""))
        kept_rows, unique_ids = count_jsonl(save_path)
        attempted = int(metadata.get("attempted_candidates") or 0)
        kept = int(metadata.get("kept_prefixes") or 0)
        rejected_refusal = int(metadata.get("rejected_refusal_candidate") or 0)
        rejected_verification = int(metadata.get("rejected_verification") or 0)
        num_examples = int(metadata.get("num_examples") or 0)
        rows.append(
            {
                "family": spec.family,
                "iteration": iteration,
                "miner": miner,
                "num_examples": num_examples,
                "attempted_candidates": attempted,
                "kept_prefixes": kept,
                "kept_jsonl_rows": kept_rows,
                "unique_prompt_ids": unique_ids,
                "kept_per_attempt": kept / attempted if attempted else 0.0,
                "kept_per_example": kept / num_examples if num_examples else 0.0,
                "refusal_reject_rate": rejected_refusal / attempted if attempted else 0.0,
                "verification_reject_rate": rejected_verification / attempted if attempted else 0.0,
                "rejected_refusal_candidate": rejected_refusal,
                "rejected_verification": rejected_verification,
                "seed_data_type": metadata.get("seed_data_type"),
                "save_path": str(save_path),
            }
        )
    return rows


def load_custom_prefix_eval_rows(spec: RunSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(spec.root.glob("iter_*/asr_eval/*.json")):
        if "final_on" not in path.name:
            continue
        with path.open() as handle:
            data = json.load(handle)
        metrics = data.get("metrics") or {}
        iteration = int(path.parent.parent.name.replace("iter_", ""))
        label = path.stem
        successes = int(metrics.get("num_success") or 0)
        total = int(metrics.get("num_tot") or 0)
        ci_low, ci_high = wilson_interval(successes, total)
        rows.append(
            {
                "family": spec.family,
                "iteration": iteration,
                "label": label,
                "asr": float(metrics.get("asr") or 0.0),
                "num_success": successes,
                "num_tot": total,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "file": str(path),
            }
        )
    return rows


def paired_counts(left: list[int], right: list[int]) -> dict[str, int]:
    both_success = sum(1 for a, b in zip(left, right) if a and b)
    left_only = sum(1 for a, b in zip(left, right) if a and not b)
    right_only = sum(1 for a, b in zip(left, right) if not a and b)
    both_safe = sum(1 for a, b in zip(left, right) if not a and not b)
    return {
        "both_success": both_success,
        "left_only": left_only,
        "right_only": right_only,
        "both_safe": both_safe,
    }


def make_family_comparisons(outcomes: list[dict[str, Any]], family_order: list[str]) -> list[dict[str, Any]]:
    if len(family_order) < 2:
        return []
    left_family, right_family = family_order[:2]
    by_key: dict[tuple[int, int, int], dict[str, int]] = {}
    for row in outcomes:
        key = (int(row["iteration"]), int(row["k"]), int(row["example_index"]))
        by_key.setdefault(key, {})[str(row["family"])] = int(row["success"])

    comparisons: list[dict[str, Any]] = []
    iteration_ks = sorted({(key[0], key[1]) for key, fams in by_key.items() if left_family in fams and right_family in fams})
    for iteration, k in iteration_ks:
        left: list[int] = []
        right: list[int] = []
        for (row_iter, row_k, _idx), fams in sorted(by_key.items()):
            if row_iter == iteration and row_k == k and left_family in fams and right_family in fams:
                left.append(fams[left_family])
                right.append(fams[right_family])
        counts = paired_counts(left, right)
        n = len(left)
        p_value = mcnemar_exact_p(counts["left_only"], counts["right_only"])
        comparisons.append(
            {
                "iteration": iteration,
                "k": k,
                "left_family": left_family,
                "right_family": right_family,
                "n": n,
                "left_asr": sum(left) / n if n else 0.0,
                "right_asr": sum(right) / n if n else 0.0,
                "right_minus_left_asr": (sum(right) - sum(left)) / n if n else 0.0,
                **counts,
                "mcnemar_exact_p": "" if p_value is None else p_value,
                "discordant_odds_left_over_right": "" if odds_ratio(counts["left_only"], counts["right_only"]) is None else odds_ratio(counts["left_only"], counts["right_only"]),
            }
        )
    return comparisons


def make_iteration_comparisons(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, int, int], int] = {}
    for row in outcomes:
        key = (str(row["family"]), int(row["iteration"]), int(row["k"]), int(row["example_index"]))
        by_key[key] = int(row["success"])
    rows: list[dict[str, Any]] = []
    families = sorted({key[0] for key in by_key})
    ks = sorted({key[2] for key in by_key})
    for family in families:
        iterations = sorted({key[1] for key in by_key if key[0] == family})
        for from_iter, to_iter in zip(iterations, iterations[1:]):
            for k in ks:
                left: list[int] = []
                right: list[int] = []
                idxs = sorted({key[3] for key in by_key if key[0] == family and key[2] == k and key[1] in {from_iter, to_iter}})
                for idx in idxs:
                    left_key = (family, from_iter, k, idx)
                    right_key = (family, to_iter, k, idx)
                    if left_key in by_key and right_key in by_key:
                        left.append(by_key[left_key])
                        right.append(by_key[right_key])
                counts = paired_counts(left, right)
                n = len(left)
                p_value = mcnemar_exact_p(counts["left_only"], counts["right_only"])
                rows.append(
                    {
                        "family": family,
                        "k": k,
                        "from_iteration": from_iter,
                        "to_iteration": to_iter,
                        "n": n,
                        "from_asr": sum(left) / n if n else 0.0,
                        "to_asr": sum(right) / n if n else 0.0,
                        "to_minus_from_asr": (sum(right) - sum(left)) / n if n else 0.0,
                        **counts,
                        "mcnemar_exact_p": "" if p_value is None else p_value,
                    }
                )
    return rows


def slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x_bar = mean(xs)
    y_bar = mean(ys)
    denom = sum((x - x_bar) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / denom


def make_iteration_envelope(asr_rows: list[AsrRow]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in sorted({row.family for row in asr_rows}):
        for iteration in sorted({row.iteration for row in asr_rows if row.family == family}):
            subset = sorted([row for row in asr_rows if row.family == family and row.iteration == iteration], key=lambda row: row.k)
            asrs = [row.asr for row in subset]
            ks = [float(row.k) for row in subset]
            successes = sum(row.num_success for row in subset)
            total = sum(row.num_tot for row in subset)
            ci_low, ci_high = wilson_interval(successes, total)
            rows.append(
                {
                    "family": family,
                    "iteration": iteration,
                    "ks": ";".join(str(row.k) for row in subset),
                    "mean_asr_across_k": mean(asrs),
                    "max_asr_across_k": max(asrs),
                    "min_asr_across_k": min(asrs),
                    "pooled_asr_across_k": successes / total if total else 0.0,
                    "pooled_num_success": successes,
                    "pooled_num_tot": total,
                    "pooled_ci95_low": ci_low,
                    "pooled_ci95_high": ci_high,
                    "asr_slope_per_token": slope(ks, asrs),
                }
            )
    return rows


def make_persistent_vulnerability_rows(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = {}
    for row in outcomes:
        key = (
            str(row["family"]),
            int(row["iteration"]),
            int(row["example_index"]),
            str(row["instruction_hash"]),
        )
        grouped.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    for (family, iteration, example_index, instruction_hash), items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: int(item["k"]))
        successful_ks = [str(item["k"]) for item in items if int(item["success"])]
        rows.append(
            {
                "family": family,
                "iteration": iteration,
                "example_index": example_index,
                "instruction_hash": instruction_hash,
                "num_k_success": len(successful_ks),
                "num_k_tested": len(items),
                "success_rate_across_k": len(successful_ks) / len(items) if items else 0.0,
                "successful_ks": ";".join(successful_ks),
            }
        )
    return rows


def make_prefix_flow_rows(specs: list[RunSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for iteration in sorted(int(path.name.replace("iter_", "")) for path in spec.root.glob("iter_*")):
            teacher_path = spec.root / f"iter_{iteration}" / f"mined_teacher_iter_{iteration}.jsonl"
            student_path = spec.root / f"iter_{iteration}" / f"mined_student_iter_{iteration}.jsonl"
            teacher_ids = load_ids(teacher_path)
            student_ids = load_ids(student_path)
            rows.append(
                {
                    "family": spec.family,
                    "iteration": iteration,
                    "teacher_unique_ids": len(teacher_ids),
                    "student_unique_ids": len(student_ids),
                    "same_iteration_overlap_ids": len(teacher_ids & student_ids),
                    "student_retention_of_teacher_ids": len(teacher_ids & student_ids) / len(teacher_ids) if teacher_ids else 0.0,
                }
            )
    return rows


def load_ids(path: Path) -> set[Any]:
    ids: set[Any] = set()
    if not path.exists():
        return ids
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            ids.add(item.get("id", item.get("instruction")))
    return ids


def fmt_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_report(
    path: Path,
    specs: list[RunSpec],
    asr_rows: list[AsrRow],
    envelope_rows: list[dict[str, Any]],
    family_comparisons: list[dict[str, Any]],
    mining_rows: list[dict[str, Any]],
    custom_rows: list[dict[str, Any]],
    persistent_rows: list[dict[str, Any]],
) -> None:
    by_family_iter = {(row["family"], int(row["iteration"])): row for row in envelope_rows}
    families = [spec.family for spec in specs]
    lines: list[str] = []
    lines.append("# Teacher-Forcing Zigzag Analysis")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    for family in families:
        first = by_family_iter.get((family, 1))
        last_iter = max(row.iteration for row in asr_rows if row.family == family)
        last = by_family_iter.get((family, last_iter))
        if first and last:
            reduction = 1 - (last["mean_asr_across_k"] / first["mean_asr_across_k"]) if first["mean_asr_across_k"] else 0.0
            lines.append(
                f"- {family}: mean static-prefix ASR across k fell from {fmt_pct(first['mean_asr_across_k'])} at iter 1 "
                f"to {fmt_pct(last['mean_asr_across_k'])} at iter {last_iter} "
                f"({fmt_pct(reduction)} relative reduction)."
            )
    if len(families) >= 2:
        last_iter = max(row.iteration for row in asr_rows)
        strongest = [row for row in family_comparisons if int(row["iteration"]) == last_iter]
        if strongest:
            lines.append(
                f"- At iter {last_iter}, {families[1]} is lower than {families[0]} on every static-prefix budget; "
                f"the largest absolute gap is {fmt_pct(abs(min(row['right_minus_left_asr'] for row in strongest)))}."
            )
    lines.append("")
    lines.append("## Static HEx-PHI Prefix ASR")
    lines.append("")
    lines.append("| family | iter | mean ASR | max ASR | pooled ASR | slope / token |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in envelope_rows:
        lines.append(
            f"| {row['family']} | {row['iteration']} | {fmt_pct(row['mean_asr_across_k'])} | "
            f"{fmt_pct(row['max_asr_across_k'])} | {fmt_pct(row['pooled_asr_across_k'])} | "
            f"{row['asr_slope_per_token']:.5f} |"
        )
    lines.append("")
    lines.append("## Persistent Prompt Failures")
    lines.append("")
    lines.append("This counts prompt indices that still jailbreak under multiple tested prefix budgets.")
    lines.append("")
    lines.append("| family | iter | >=1 k | >=2 k | all k |")
    lines.append("|---|---:|---:|---:|---:|")
    for family in families:
        iterations = sorted({int(row["iteration"]) for row in persistent_rows if row["family"] == family})
        for iteration in iterations:
            subset = [row for row in persistent_rows if row["family"] == family and int(row["iteration"]) == iteration]
            at_least_1 = sum(int(row["num_k_success"]) >= 1 for row in subset)
            at_least_2 = sum(int(row["num_k_success"]) >= 2 for row in subset)
            all_k = sum(int(row["num_k_success"]) == int(row["num_k_tested"]) for row in subset)
            lines.append(f"| {family} | {iteration} | {at_least_1} | {at_least_2} | {all_k} |")
    lines.append("")
    lines.append("## Cross-Family Paired Comparison")
    lines.append("")
    lines.append("Exact McNemar tests compare matched prompt indices under the same iteration and prefix budget.")
    lines.append("")
    lines.append("| iter | k | left ASR | right ASR | right-left | left-only | right-only | p |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in family_comparisons:
        p_value = row["mcnemar_exact_p"]
        p_str = "" if p_value == "" else f"{float(p_value):.3g}"
        lines.append(
            f"| {row['iteration']} | {row['k']} | {fmt_pct(row['left_asr'])} | {fmt_pct(row['right_asr'])} | "
            f"{fmt_pct(row['right_minus_left_asr'])} | {row['left_only']} | {row['right_only']} | {p_str} |"
        )
    lines.append("")
    lines.append("## Mining Yield")
    lines.append("")
    lines.append("| family | iter | miner | kept / attempted | kept / example | unique prompt IDs | refusal rejects | verification rejects |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for row in sorted(mining_rows, key=lambda r: (r["family"], int(r["iteration"]), r["miner"])):
        lines.append(
            f"| {row['family']} | {row['iteration']} | {row['miner']} | {fmt_pct(row['kept_per_attempt'])} | "
            f"{row['kept_per_example']:.2f} | {row['unique_prompt_ids']} | "
            f"{fmt_pct(row['refusal_reject_rate'])} | {fmt_pct(row['verification_reject_rate'])} |"
        )
    if custom_rows:
        lines.append("")
        lines.append("## Custom Mined-Prefix Eval")
        lines.append("")
        lines.append("These are evals on mined prefix sets, not the full static HEx-PHI harmful-prefix sweep.")
        lines.append("")
        lines.append("| family | iter | label | ASR | successes / total | 95% CI |")
        lines.append("|---|---:|---|---:|---:|---:|")
        for row in custom_rows:
            lines.append(
                f"| {row['family']} | {row['iteration']} | {row['label']} | {fmt_pct(row['asr'])} | "
                f"{row['num_success']} / {row['num_tot']} | "
                f"[{fmt_pct(row['ci95_low'])}, {fmt_pct(row['ci95_high'])}] |"
            )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- Safety success here is the existing keyword-jailbreak metric, so the report is best read as a structured robustness screen rather than a final semantic safety judgment.")
    lines.append("- The paired tests use matched row indices from the saved eval files. They do not assume independent prompts, which makes them more appropriate than comparing only aggregate ASR.")
    lines.append("- Custom mined-prefix evals can be high even when static-prefix ASR is low; they measure a different, adversarially selected distribution.")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    specs = parse_run_specs(args.run)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    asr_rows = [row for spec in specs for row in load_asr_rows(spec)]

    asr_table = []
    for row in asr_rows:
        ci_low, ci_high = wilson_interval(row.num_success, row.num_tot)
        asr_table.append(
            {
                "family": row.family,
                "iteration": row.iteration,
                "k": row.k,
                "bench": row.bench,
                "asr": row.asr,
                "num_success": row.num_success,
                "num_tot": row.num_tot,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "file": str(row.eval_path),
            }
        )

    outcomes = [outcome for row in asr_rows for outcome in load_eval_outcomes(row)]
    envelope_rows = make_iteration_envelope(asr_rows)
    family_comparisons = make_family_comparisons(outcomes, [spec.family for spec in specs])
    iteration_comparisons = make_iteration_comparisons(outcomes)
    mining_rows = [row for spec in specs for row in load_mining_rows(spec)]
    prefix_flow_rows = make_prefix_flow_rows(specs)
    custom_rows = [row for spec in specs for row in load_custom_prefix_eval_rows(spec)]
    persistent_rows = make_persistent_vulnerability_rows(outcomes)

    write_csv(
        output_dir / "asr_with_confidence.csv",
        asr_table,
        ["family", "iteration", "k", "bench", "asr", "num_success", "num_tot", "ci95_low", "ci95_high", "file"],
    )
    write_csv(
        output_dir / "eval_outcomes.csv",
        outcomes,
        ["family", "iteration", "k", "example_index", "instruction_hash", "success", "output_chars"],
    )
    write_csv(
        output_dir / "iteration_envelope.csv",
        envelope_rows,
        [
            "family",
            "iteration",
            "ks",
            "mean_asr_across_k",
            "max_asr_across_k",
            "min_asr_across_k",
            "pooled_asr_across_k",
            "pooled_num_success",
            "pooled_num_tot",
            "pooled_ci95_low",
            "pooled_ci95_high",
            "asr_slope_per_token",
        ],
    )
    write_csv(
        output_dir / "family_paired_comparison.csv",
        family_comparisons,
        [
            "iteration",
            "k",
            "left_family",
            "right_family",
            "n",
            "left_asr",
            "right_asr",
            "right_minus_left_asr",
            "both_success",
            "left_only",
            "right_only",
            "both_safe",
            "mcnemar_exact_p",
            "discordant_odds_left_over_right",
        ],
    )
    write_csv(
        output_dir / "iteration_paired_changes.csv",
        iteration_comparisons,
        [
            "family",
            "k",
            "from_iteration",
            "to_iteration",
            "n",
            "from_asr",
            "to_asr",
            "to_minus_from_asr",
            "both_success",
            "left_only",
            "right_only",
            "both_safe",
            "mcnemar_exact_p",
        ],
    )
    write_csv(
        output_dir / "mining_yield_summary.csv",
        mining_rows,
        [
            "family",
            "iteration",
            "miner",
            "num_examples",
            "attempted_candidates",
            "kept_prefixes",
            "kept_jsonl_rows",
            "unique_prompt_ids",
            "kept_per_attempt",
            "kept_per_example",
            "refusal_reject_rate",
            "verification_reject_rate",
            "rejected_refusal_candidate",
            "rejected_verification",
            "seed_data_type",
            "save_path",
        ],
    )
    write_csv(
        output_dir / "prefix_id_flow.csv",
        prefix_flow_rows,
        [
            "family",
            "iteration",
            "teacher_unique_ids",
            "student_unique_ids",
            "same_iteration_overlap_ids",
            "student_retention_of_teacher_ids",
        ],
    )
    write_csv(
        output_dir / "persistent_vulnerability_by_example.csv",
        persistent_rows,
        [
            "family",
            "iteration",
            "example_index",
            "instruction_hash",
            "num_k_success",
            "num_k_tested",
            "success_rate_across_k",
            "successful_ks",
        ],
    )
    write_csv(
        output_dir / "custom_prefix_eval_summary.csv",
        custom_rows,
        ["family", "iteration", "label", "asr", "num_success", "num_tot", "ci95_low", "ci95_high", "file"],
    )
    write_report(
        output_dir / "report.md",
        specs,
        asr_rows,
        envelope_rows,
        family_comparisons,
        mining_rows,
        custom_rows,
        persistent_rows,
    )

    print(json.dumps({"output_dir": str(output_dir), "rows": {"asr": len(asr_table), "outcomes": len(outcomes), "mining": len(mining_rows)}}, indent=2))


if __name__ == "__main__":
    main()
