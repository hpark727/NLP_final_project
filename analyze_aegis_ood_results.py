from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


EVAL_DIR = Path("logs/final_adaptive_ood_package/aegis_eval")
PREFIX_PATH = Path("logs/final_adaptive_ood_package/aegis/aegis_unsafe_prefixes_500.jsonl")
OUT_DIR = Path("logs/final_adaptive_ood_package")
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
REPORT_PATH = OUT_DIR / "final_adaptive_ood_report.md"

MODEL_ORDER = ["original_augmented", "adaptive_only_recovery", "mixed_50_50_recovery"]
MODEL_LABELS = {
    "original_augmented": "Original augmented",
    "adaptive_only_recovery": "Adaptive-only recovery",
    "mixed_50_50_recovery": "Mixed 50/50 recovery",
}
MODEL_COLORS = {
    "original_augmented": "#6d6d6d",
    "adaptive_only_recovery": "#b94633",
    "mixed_50_50_recovery": "#2f7f5f",
}


def split_categories(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return ["Unspecified"]
    return [part.strip() for part in value.split(",") if part.strip()] or ["Unspecified"]


def is_success(item: list[dict[str, Any]]) -> bool:
    # eval_safety_tinker's keyword metric stores the final generated continuation
    # as the last assistant message.
    from finetuning_buckets.inference.safety_eval.keyword_eval import KeyWordEvaluator

    return KeyWordEvaluator.is_jailbroken(str(item[-1]["content"]))


def load_prefix_records() -> list[dict[str, Any]]:
    return [json.loads(line) for line in PREFIX_PATH.read_text().splitlines() if line.strip()]


def load_eval(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = load_prefix_records()
    rows = []
    per_example: dict[tuple[str, int], list[bool]] = {}

    for model in MODEL_ORDER:
        for k in [5, 10, 20, 40]:
            path = EVAL_DIR / f"{model}_aegis_k{k}.json"
            data = load_eval(path)
            metrics = data["metrics"]
            successes = [is_success(item) for item in data["results"]]
            per_example[(model, k)] = successes
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "k": k,
                    "n": metrics["num_tot"],
                    "success": metrics["num_success"],
                    "asr": metrics["asr"],
                    "file": str(path),
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(TABLE_DIR / "aegis_ood_summary.csv", index=False)

    category_rows = []
    for model in MODEL_ORDER:
        for k in [5, 10, 20, 40]:
            successes = per_example[(model, k)]
            counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            for record, success in zip(records, successes):
                for category in split_categories(record.get("category")):
                    counts[category][1] += 1
                    counts[category][0] += int(success)
            for category, (success, n) in counts.items():
                if n >= 10:
                    category_rows.append(
                        {
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "k": k,
                            "category": category,
                            "success": success,
                            "n": n,
                            "asr": success / n if n else 0.0,
                        }
                    )
    category_summary = pd.DataFrame(category_rows)
    category_summary.to_csv(TABLE_DIR / "aegis_ood_category_summary.csv", index=False)

    paired_rows = []
    for k in [5, 10, 20, 40]:
        for left, right in [
            ("original_augmented", "mixed_50_50_recovery"),
            ("adaptive_only_recovery", "mixed_50_50_recovery"),
            ("original_augmented", "adaptive_only_recovery"),
        ]:
            left_s = per_example[(left, k)]
            right_s = per_example[(right, k)]
            both = sum(l and r for l, r in zip(left_s, right_s))
            left_only = sum(l and not r for l, r in zip(left_s, right_s))
            right_only = sum(r and not l for l, r in zip(left_s, right_s))
            neither = sum(not l and not r for l, r in zip(left_s, right_s))
            paired_rows.append(
                {
                    "k": k,
                    "left": left,
                    "right": right,
                    "left_asr": sum(left_s) / len(left_s),
                    "right_asr": sum(right_s) / len(right_s),
                    "right_minus_left": sum(right_s) / len(right_s) - sum(left_s) / len(left_s),
                    "both_success": both,
                    "left_only": left_only,
                    "right_only": right_only,
                    "neither": neither,
                    "discordant": left_only + right_only,
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(TABLE_DIR / "aegis_ood_paired_overlap.csv", index=False)
    return summary, category_summary, paired


def style_axes(ax, title: str, ylabel: str | None = None, xlabel: str | None = None) -> None:
    ax.set_title(title, fontsize=13, weight="bold", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_summary(summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model in MODEL_ORDER:
        sub = summary[summary["model"] == model].sort_values("k")
        ax.plot(
            sub["k"],
            sub["asr"] * 100,
            marker="o",
            linewidth=2.5,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    style_axes(ax, "Aegis cross-dataset unsafe-prefix stress test", "Keyword ASR (%)", "Prefill length k")
    ax.set_xticks([5, 10, 20, 40])
    ax.legend(frameon=False)
    path = FIG_DIR / "10_aegis_ood_asr.png"
    savefig(path)
    return path


def plot_delta(summary: pd.DataFrame) -> Path:
    pivot = summary.pivot_table(index="k", columns="model", values="asr").sort_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(
        pivot.index,
        (pivot["mixed_50_50_recovery"] - pivot["adaptive_only_recovery"]) * 100,
        marker="o",
        linewidth=2.5,
        color="#2f7f5f",
        label="Mixed minus adaptive-only",
    )
    ax.plot(
        pivot.index,
        (pivot["mixed_50_50_recovery"] - pivot["original_augmented"]) * 100,
        marker="o",
        linewidth=2.5,
        color="#5f6c7b",
        label="Mixed minus original augmented",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    style_axes(ax, "Aegis deltas clarify what mixed recovery does and does not solve", "ASR delta (percentage points)", "Prefill length k")
    ax.set_xticks([5, 10, 20, 40])
    ax.legend(frameon=False)
    path = FIG_DIR / "11_aegis_ood_deltas.png"
    savefig(path)
    return path


def plot_category(category_summary: pd.DataFrame) -> Path:
    k = 40
    rows = category_summary[category_summary["k"] == k].copy()
    # Show categories with enough examples and largest adaptive-only ASR.
    adaptive = rows[rows["model"] == "adaptive_only_recovery"].sort_values("asr", ascending=False)
    keep = adaptive.head(10)["category"].tolist()
    rows = rows[rows["category"].isin(keep)]
    pivot = rows.pivot_table(index="category", columns="model", values="asr")
    pivot = pivot.loc[keep[::-1]]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    y = range(len(pivot))
    height = 0.24
    for offset, model in [(-height, "original_augmented"), (0, "adaptive_only_recovery"), (height, "mixed_50_50_recovery")]:
        ax.barh(
            [i + offset for i in y],
            pivot[model] * 100,
            height=height,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    ax.set_yticks(list(y))
    ax.set_yticklabels(pivot.index)
    style_axes(ax, "Aegis k=40 category ASR: mixed improves adaptive-only broadly", "ASR (%)")
    ax.legend(frameon=False, fontsize=8)
    path = FIG_DIR / "12_aegis_ood_category_k40.png"
    savefig(path)
    return path


def update_report(summary: pd.DataFrame, paired: pd.DataFrame, figs: list[Path]) -> None:
    pivot = summary.pivot_table(index="k", columns="model", values="asr").sort_index()
    successes = summary.pivot_table(index="k", columns="model", values="success").sort_index()
    n = int(summary["n"].iloc[0])
    k40_mixed_vs_adaptive = paired[
        (paired["k"] == 40)
        & (paired["left"] == "adaptive_only_recovery")
        & (paired["right"] == "mixed_50_50_recovery")
    ].iloc[0]
    k40_mixed_vs_original = paired[
        (paired["k"] == 40)
        & (paired["left"] == "original_augmented")
        & (paired["right"] == "mixed_50_50_recovery")
    ].iloc[0]

    section = f"""

## Aegis Cross-Dataset Results

The Aegis run is complete on the 500-example unsafe prompt + unsafe response sample. This tests a stronger OOD setting than the main HEx-PHI results because both the harmful prompt source and the unsafe prefill source differ from the recovery data.

![Aegis OOD ASR](figures/10_aegis_ood_asr.png)

| k | Original augmented | Adaptive-only recovery | Mixed 50/50 recovery |
|---:|---:|---:|---:|
"""
    for k, row in pivot.iterrows():
        section += (
            f"| {int(k)} | "
            f"{successes.loc[k, 'original_augmented']:.0f}/{n} = {row['original_augmented']*100:.1f}% | "
            f"{successes.loc[k, 'adaptive_only_recovery']:.0f}/{n} = {row['adaptive_only_recovery']*100:.1f}% | "
            f"{successes.loc[k, 'mixed_50_50_recovery']:.0f}/{n} = {row['mixed_50_50_recovery']*100:.1f}% |\n"
        )

    section += f"""

Interpretation:

- Mixed recovery strongly improves over adaptive-only recovery on Aegis: at `k=40`, ASR drops from `{pivot.loc[40, 'adaptive_only_recovery']*100:.1f}%` to `{pivot.loc[40, 'mixed_50_50_recovery']*100:.1f}%`.
- Original augmented remains best on this Aegis sample: at `k=40`, it has `{pivot.loc[40, 'original_augmented']*100:.1f}%` ASR versus mixed recovery's `{pivot.loc[40, 'mixed_50_50_recovery']*100:.1f}%`.
- This means Aegis strengthens the "adaptive-only forgets; mixed restores much of the lost robustness" claim, but it does not support claiming that mixed recovery dominates all prompt-source OOD settings.
- Paired at `k=40`, mixed succeeds on `{int(k40_mixed_vs_adaptive['right_only'])}` examples where adaptive-only does not, while adaptive-only succeeds on `{int(k40_mixed_vs_adaptive['left_only'])}` examples where mixed does not.
- Paired at `k=40`, mixed succeeds on `{int(k40_mixed_vs_original['right_only'])}` examples where original augmented does not, while original augmented succeeds on `{int(k40_mixed_vs_original['left_only'])}` examples where mixed does not.

![Aegis OOD deltas](figures/11_aegis_ood_deltas.png)

![Aegis category ASR](figures/12_aegis_ood_category_k40.png)

Revised OOD claim:

> Mixed recovery is robust to prefix-source OOD and partially improves prompt-source OOD relative to adaptive-only recovery, but the Aegis results show that adding adaptive recovery can still trade off against the original augmented model on some cross-dataset unsafe-prefix distributions.
"""

    text = REPORT_PATH.read_text()
    marker = "\n## Recommended Final Claim\n"
    if "## Aegis Cross-Dataset Results" in text:
        start = text.index("\n## Aegis Cross-Dataset Results")
        end = text.index(marker)
        text = text[:start] + section + text[end:]
    else:
        text = text.replace(marker, section + marker)
    REPORT_PATH.write_text(text)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})
    summary, category_summary, paired = build_tables()
    figs = [plot_summary(summary), plot_delta(summary), plot_category(category_summary)]
    update_report(summary, paired, figs)
    print(
        json.dumps(
            {
                "summary_csv": str(TABLE_DIR / "aegis_ood_summary.csv"),
                "category_csv": str(TABLE_DIR / "aegis_ood_category_summary.csv"),
                "paired_csv": str(TABLE_DIR / "aegis_ood_paired_overlap.csv"),
                "figures": [str(p) for p in figs],
                "report": str(REPORT_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
