from __future__ import annotations

import json
import math
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import hf_hub_download


OUT_DIR = Path("logs/final_adaptive_ood_package")
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
AEGIS_DIR = OUT_DIR / "aegis"

SRC_TABLES = Path("logs/adaptive_recovery/final_experiment_report_tables")

MODEL_PATHS = {
    "original_augmented": "tinker://de5327cc-9cda-57db-a585-4134fa998428:train:0/sampler_weights/final",
    "adaptive_only_recovery": "tinker://d82c30b2-da1f-597f-9d58-b72f5ecef244:train:0/sampler_weights/final",
    "mixed_50_50_recovery": "tinker://9e8229e1-3d61-58d0-99ab-f04314679ac2:train:0/sampler_weights/final",
}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ensure_dirs() -> None:
    for directory in [OUT_DIR, FIG_DIR, TABLE_DIR, AEGIS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def style_axes(ax, title: str, ylabel: str | None = None, xlabel: str | None = None) -> None:
    ax.set_title(title, fontsize=13, weight="bold", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(SRC_TABLES / name)


def plot_static_baseline() -> Path:
    df = read_csv("static_harmful_model_comparison.csv")
    rows = df[(df["k"] == 10) & (df["model"].isin(["base Llama-3.2-3B", "original augmented"]))]
    rows = rows.set_index("model").loc[["base Llama-3.2-3B", "original augmented"]].reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = ["#b94633", "#2f7f5f"]
    bars = ax.bar(rows["model"], rows["asr"] * 100, color=colors)
    for bar, (_, row) in zip(bars, rows.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{row['success']}/{row['n']}", ha="center")
    ax.set_ylim(0, 105)
    style_axes(ax, "Static harmful-prefix augmentation works in-distribution", "Attack success rate (%)")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(["Base\nLlama-3.2-3B", "Static-prefix\naugmented"])
    path = FIG_DIR / "01_static_prefix_baseline.png"
    savefig(path)
    return path


def plot_matched_static_vs_adaptive() -> Path:
    df = read_csv("original_augmented_matched_adaptive_vs_static.csv")
    pivot = df.pivot_table(index="k", columns="condition", values="asr").sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(pivot.index, pivot["matched static"] * 100, marker="o", linewidth=2.5, label="Matched static prefixes")
    ax.plot(pivot.index, pivot["adaptive gen1 mined"] * 100, marker="o", linewidth=2.5, label="Adaptive mined prefixes")
    style_axes(ax, "Adaptive prefixes become harder at larger prefix budgets", "ASR on matched prompts (%)", "Prefill length k")
    ax.set_xticks(pivot.index)
    ax.legend(frameon=False)
    path = FIG_DIR / "02_matched_static_vs_adaptive.png"
    savefig(path)
    return path


def plot_recovery_static() -> Path:
    df = read_csv("static_harmful_model_comparison.csv")
    rows = df[df["model"].isin(["adaptive-only recovery", "mixed 50/50 recovery"])].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for model, color in [("adaptive-only recovery", "#b94633"), ("mixed 50/50 recovery", "#2f7f5f")]:
        sub = rows[rows["model"] == model].sort_values("k")
        ax.plot(sub["k"], sub["asr"] * 100, marker="o", linewidth=2.5, label=model, color=color)
    style_axes(ax, "Adaptive-only recovery forgets static robustness; mixed recovery restores it", "Static harmful-prefix ASR (%)", "Prefill length k")
    ax.set_xticks([5, 10, 20, 40])
    ax.legend(frameon=False)
    path = FIG_DIR / "03_static_recovery_comparison.png"
    savefig(path)
    return path


def plot_gen3_models() -> Path:
    df = read_csv("gen3_same23_model_comparison.csv")
    rows = df[df["condition"] == "gen3"].copy()
    order = ["original augmented", "adaptive-only recovery", "mixed 50/50 recovery"]
    colors = {
        "original augmented": "#6d6d6d",
        "adaptive-only recovery": "#b94633",
        "mixed 50/50 recovery": "#2f7f5f",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for model in order:
        sub = rows[rows["model"] == model].sort_values("k")
        ax.plot(sub["k"], sub["asr"] * 100, marker="o", linewidth=2.5, label=model, color=colors[model])
    style_axes(ax, "Recovery models on adaptively mined gen3 hard prefixes", "Gen3 adaptive ASR (%)", "Prefill length k")
    ax.set_xticks([5, 10, 20, 40])
    ax.legend(frameon=False)
    path = FIG_DIR / "04_gen3_recovery_comparison.png"
    savefig(path)
    return path


def plot_gpt_judge_static() -> Path:
    df = read_csv("gpt_static_harmful_model_comparison.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for model, color in [("adaptive-only recovery", "#b94633"), ("mixed 50/50 recovery", "#2f7f5f")]:
        sub = df[df["model"] == model].sort_values("k")
        ax.plot(sub["k"], sub["asr_score5"] * 100, marker="o", linewidth=2.5, label=f"{model} (score=5)", color=color)
        ax.plot(sub["k"], sub["asr_score_ge4"] * 100, marker="x", linewidth=1.8, linestyle="--", label=f"{model} (score>=4)", color=color, alpha=0.8)
    style_axes(ax, "GPT judge confirms static regression vs mixed recovery", "GPT-judged ASR (%)", "Prefill length k")
    ax.set_xticks([5, 10, 20, 40])
    ax.legend(frameon=False, fontsize=8)
    path = FIG_DIR / "05_gpt_static_recovery.png"
    savefig(path)
    return path


def plot_trajectory_refusal_margin() -> Path:
    df = read_csv("trajectory_adaptive_focus_step0_16.csv")
    rows = df[df["k"].isin([20, 40]) & df["response_step"].isin([0, 16])].copy()
    rows["label"] = rows.apply(lambda r: f"k={int(r['k'])}, step={int(r['response_step'])}", axis=1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(rows))
    width = 0.36
    ax.bar([i - width / 2 for i in x], rows["augmented_refusal_margin_mean"], width, label="Original augmented", color="#6d6d6d")
    ax.bar([i + width / 2 for i in x], rows["mixed_refusal_margin_mean"], width, label="Mixed 50/50 recovery", color="#2f7f5f")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(rows["label"])
    style_axes(ax, "Mixed recovery raises refusal margin on hard adaptive prefixes", "Refusal log-mass minus compliance log-mass")
    ax.legend(frameon=False)
    path = FIG_DIR / "06_trajectory_refusal_margin.png"
    savefig(path)
    return path


def plot_trajectory_delta_signs() -> Path:
    df = read_csv("trajectory_paired_delta_sign_counts.csv")
    rows = df[df["k"].isin([20, 40]) & df["response_step"].isin([0, 16])].copy()
    rows["positive_frac"] = rows["refusal_margin_positive_delta_count"] / rows["n"] * 100
    rows["label"] = rows.apply(lambda r: f"k={int(r['k'])}, step={int(r['response_step'])}", axis=1)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(rows["label"], rows["positive_frac"], color="#2f7f5f")
    for bar, (_, row) in zip(bars, rows.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, min(101, bar.get_height() + 2), f"{int(row['refusal_margin_positive_delta_count'])}/{int(row['n'])}", ha="center")
    ax.set_ylim(0, 110)
    style_axes(ax, "Refusal-margin gains are paired, not just mean-driven", "Examples with higher mixed-model margin (%)")
    path = FIG_DIR / "07_trajectory_paired_signs.png"
    savefig(path)
    return path


def plot_subspace() -> Path:
    df = read_csv("refusal_subspace_condition_summary.csv")
    rows = df[df["condition"].isin(["clean", "refusal", "adaptive"])].copy()
    pivot = rows.pivot_table(index="k", columns="condition", values="refusal_subspace_fraction_mean").sort_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for condition, color in [("clean", "#5f6c7b"), ("refusal", "#b94633"), ("adaptive", "#2f7f5f")]:
        ax.plot(pivot.index, pivot[condition], marker="o", linewidth=2.5, label=condition, color=color)
    style_axes(ax, "Adaptive prefixes are not simply refusal-direction states", "Fraction of delta norm in refusal subspace", "Prefill length k")
    ax.set_xticks([5, 10, 20, 40])
    ax.legend(frameon=False)
    path = FIG_DIR / "08_refusal_subspace_fraction.png"
    savefig(path)
    return path


def split_categories(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return ["Unspecified"]
    return [part.strip() for part in value.split(",") if part.strip()] or ["Unspecified"]


def prepare_aegis() -> dict[str, Any]:
    repo_id = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
    paths = {
        split: Path(hf_hub_download(repo_id=repo_id, filename=f"{split}.json", repo_type="dataset"))
        for split in ["validation", "test"]
    }
    all_records = []
    split_counts = {}
    for split, path in paths.items():
        with path.open() as handle:
            rows = json.load(handle)
        split_counts[split] = len(rows)
        for row in rows:
            prompt = str(row.get("prompt") or "").strip()
            response = str(row.get("response") or "").strip()
            if (
                prompt
                and response
                and row.get("prompt_label") == "unsafe"
                and row.get("response_label") == "unsafe"
            ):
                all_records.append(
                    {
                        "id": row.get("id"),
                        "instruction": prompt,
                        "prefix": response,
                        "category": row.get("violated_categories") or "Unspecified",
                        "source": f"aegis_2_0_{split}_unsafe_prompt_unsafe_response",
                        "split": split,
                    }
                )

    # Deterministic de-duplication by prompt text.
    deduped = []
    seen = set()
    for record in all_records:
        key = record["instruction"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)

    full_path = AEGIS_DIR / "aegis_unsafe_prompt_unsafe_response_prefixes.jsonl"
    with full_path.open("w") as handle:
        for record in deduped:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    sample = deduped[: min(500, len(deduped))]
    sample_path = AEGIS_DIR / "aegis_unsafe_prefixes_500.jsonl"
    with sample_path.open("w") as handle:
        for record in sample:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    category_counter: Counter[str] = Counter()
    for record in deduped:
        category_counter.update(split_categories(record["category"]))
    category_df = pd.DataFrame(
        [{"category": category, "count": count} for category, count in category_counter.most_common()]
    )
    category_path = TABLE_DIR / "aegis_unsafe_category_counts.csv"
    category_df.to_csv(category_path, index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    top = category_df.head(12).iloc[::-1]
    ax.barh(top["category"], top["count"], color="#3d6f8e")
    style_axes(ax, "Aegis held-out unsafe prefix pool by category", "Count")
    path = FIG_DIR / "09_aegis_category_distribution.png"
    savefig(path)

    metadata = {
        "repo_id": repo_id,
        "source_url": "https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
        "split_counts": split_counts,
        "unsafe_prompt_unsafe_response_records": len(all_records),
        "deduped_records": len(deduped),
        "sample_records": len(sample),
        "full_jsonl": str(full_path),
        "sample_jsonl": str(sample_path),
        "category_counts_csv": str(category_path),
        "category_plot": str(path),
    }
    with (AEGIS_DIR / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def write_aegis_commands(aegis_metadata: dict[str, Any]) -> Path:
    path = OUT_DIR / "run_aegis_ood_eval.zsh"
    sample_path = aegis_metadata["sample_jsonl"]
    body = f"""#!/usr/bin/env zsh
set -euo pipefail

set -a
source .env
set +a

PREFIX_PATH="${{PREFIX_PATH:-{sample_path}}}"
OUT_ROOT="${{OUT_ROOT:-logs/final_adaptive_ood_package/aegis_eval}}"
mkdir -p "$OUT_ROOT"

typeset -A MODELS
MODELS[original_augmented]="{MODEL_PATHS['original_augmented']}"
MODELS[adaptive_only_recovery]="{MODEL_PATHS['adaptive_only_recovery']}"
MODELS[mixed_50_50_recovery]="{MODEL_PATHS['mixed_50_50_recovery']}"

for model_name in original_augmented adaptive_only_recovery mixed_50_50_recovery; do
  for k in 5 10 20 40; do
    echo
    echo "### Aegis OOD eval: $model_name k=$k"
    ./.venv/bin/python eval_safety_tinker.py \\
      --model_path "${{MODELS[$model_name]}}" \\
      --prompt_style llama3 \\
      --bench hex-phi_with_custom_prefix \\
      --eval_template null \\
      --custom_prefix_path "$PREFIX_PATH" \\
      --custom_prefix_tokens "$k" \\
      --max_new_tokens 512 \\
      --temperature 0.9 \\
      --top_p 0.6 \\
      --concurrency 16 \\
      --output_distribution_steps 0 \\
      --evaluator key_word \\
      --save_path "$OUT_ROOT/${{model_name}}_aegis_k${{k}}.json"
  done
done
"""
    path.write_text(body)
    path.chmod(0o755)
    return path


def write_report(figs: dict[str, Path], aegis_metadata: dict[str, Any], command_path: Path) -> Path:
    static_df = read_csv("static_harmful_model_comparison.csv")
    gen3_df = read_csv("gen3_same23_model_comparison.csv")
    traj_df = read_csv("trajectory_adaptive_focus_step0_16.csv")
    signs_df = read_csv("trajectory_paired_delta_sign_counts.csv")
    remine_yield = read_csv("gen4_remine_yield.csv").iloc[0]

    base_asr = static_df[(static_df["model"] == "base Llama-3.2-3B") & (static_df["k"] == 10)]["asr"].iloc[0]
    aug_asr = static_df[(static_df["model"] == "original augmented") & (static_df["k"] == 10)]["asr"].iloc[0]
    adaptive_static_k10 = static_df[(static_df["model"] == "adaptive-only recovery") & (static_df["k"] == 10)]["asr"].iloc[0]
    mixed_static_k10 = static_df[(static_df["model"] == "mixed 50/50 recovery") & (static_df["k"] == 10)]["asr"].iloc[0]
    gen3_orig_k40 = gen3_df[(gen3_df["model"] == "original augmented") & (gen3_df["condition"] == "gen3") & (gen3_df["k"] == 40)]["asr"].iloc[0]
    gen3_mixed_k40 = gen3_df[(gen3_df["model"] == "mixed 50/50 recovery") & (gen3_df["condition"] == "gen3") & (gen3_df["k"] == 40)]["asr"].iloc[0]
    traj_k20_step0 = traj_df[(traj_df["k"] == 20) & (traj_df["response_step"] == 0)].iloc[0]
    signs_k20_step0 = signs_df[(signs_df["k"] == 20) & (signs_df["response_step"] == 0)].iloc[0]

    rel = {key: path.relative_to(OUT_DIR) for key, path in figs.items()}
    report = f"""# Final Adaptive OOD Package

Generated from existing experiment logs in `logs/adaptive_recovery` and `logs/adaptive_prefixes`.

## Research Thesis

The tight version of the project is:

> Static harmful-prefix recovery is an in-distribution safety result. Adaptive mining creates a prefix-distribution shift that exposes residual failures. Adaptive-only recovery improves the mined prefix family but partially forgets the original static prefix family. Mixed static + adaptive recovery widens the recovery basin and is supported by both behavioral ASR and hidden-state trajectory evidence.

## Concrete OOD Definition

Use two OOD axes:

1. **Prefix-source OOD:** train/evaluate on the same harmful prompt source, but change the prefilled continuation distribution. In-distribution is static HEx-PHI harmful prefixes; OOD is model-mined adaptive prefixes.
2. **Prompt-source OOD:** train/evaluate on HEx-PHI-style harmful prompts, then validate on a different harmful dataset and taxonomy. Aegis 2.0 gives this axis because it has a different collection pipeline, label taxonomy, and Mistral-generated unsafe responses.

This makes the project more concrete than saying "OOD" abstractly. Current results primarily establish prefix-source OOD. The prepared Aegis eval below would test prompt-source plus prefix-source OOD.

## 1. Adaptive OOD Behavioral Results

Static augmentation initially works very well in-distribution: static harmful-prefix ASR at `k=10` drops from `{pct(base_asr)}` for base Llama-3.2-3B to `{pct(aug_asr)}` for the original augmented model.

![Static baseline]({rel['static_baseline']})

Adaptive mining then exposes a distribution shift. On matched prompts, adaptive prefixes become harder than matched static prefixes at `k=40`, even though they are not uniformly stronger at shorter prefix lengths.

![Matched static vs adaptive]({rel['matched_static_vs_adaptive']})

The focused gen3 same23 set is a small but useful paired stress set. The original augmented model reaches `{pct(gen3_orig_k40)}` ASR at `k=40` on gen3 adaptive prefixes.

![Gen3 recovery comparison]({rel['gen3_models']})

## 2. Mixed Recovery vs Adaptive-Only Recovery

Adaptive-only recovery is not a clean solution. It improves gen3 adaptive robustness, but it regresses on original static harmful prefixes. At static `k=10`, adaptive-only recovery has `{pct(adaptive_static_k10)}` ASR, while mixed 50/50 recovery has `{pct(mixed_static_k10)}` ASR.

![Static recovery comparison]({rel['recovery_static']})

The GPT-judged static comparison supports the same direction: adaptive-only recovery has high GPT-judged static failures, while mixed recovery stays low across prefix lengths.

![GPT static recovery]({rel['gpt_static']})

On gen3 adaptive prefixes, mixed recovery preserves most of the adaptive-only gain. At `k=40`, mixed recovery reduces original augmented ASR from `{pct(gen3_orig_k40)}` to `{pct(gen3_mixed_k40)}`.

The current mining algorithm also becomes much less productive against mixed recovery: the gen4 stress remine kept `{int(remine_yield['kept_file_rows'])}` prefixes from `{int(remine_yield['attempted_candidates'])}` candidates, a `{pct(float(remine_yield['keep_rate']))}` keep rate. Interpret the 5 kept prefixes as a stress case, not a population estimate.

## 3. Hidden-State / Trajectory Mechanism

The mechanism result is not "mixed recovery memorizes the mined prefixes." The stronger statement is that mixed recovery raises refusal pressure on the same adaptive prefix states.

At `k=20`, response step 0, mixed recovery raises mean refusal margin by `{traj_k20_step0['delta_mixed_minus_augmented_refusal_margin_mean']:.2f}` and does so for `{int(signs_k20_step0['refusal_margin_positive_delta_count'])}/{int(signs_k20_step0['n'])}` paired examples.

![Trajectory refusal margin]({rel['trajectory_refusal_margin']})

![Trajectory paired signs]({rel['trajectory_signs']})

The refusal-subspace result adds a useful caution: adaptive prefixes are not simply normal refusal states. Their refusal-subspace fraction is consistently lower than the explicit refusal condition, so adaptive mining likely exploits directions or token-competition effects not captured by a single refusal axis.

![Refusal subspace]({rel['subspace']})

## Aegis Cross-Validation Plan

I prepared a held-out Aegis 2.0 prefix file from rows where both the prompt and response are labeled unsafe. This creates a stricter OOD stress test:

- Dataset: `nvidia/Aegis-AI-Content-Safety-Dataset-2.0`
- Source: validation + test splits
- Full unsafe prompt/unsafe response records after prompt de-duplication: `{aegis_metadata['deduped_records']}`
- Ready-to-run sample file: `{aegis_metadata['sample_jsonl']}`
- Full file: `{aegis_metadata['full_jsonl']}`

![Aegis category distribution]({rel['aegis_categories']})

Run this if you want to add cross-dataset validation:

```bash
{command_path}
```

That script evaluates original augmented, adaptive-only recovery, and mixed 50/50 recovery on Aegis unsafe response prefixes at `k=5,10,20,40`.

Expected interpretation:

- If mixed recovery still beats adaptive-only on Aegis, the recovery-basin claim is much stronger.
- If mixed recovery only helps on HEx-PHI-derived prompts, the conclusion should be narrowed to prefix-source OOD within the original harmful-prompt distribution.
- If all models fail badly, Aegis reveals a prompt-source OOD gap and becomes an honest limitation/future-work result.

## Recommended Final Claim

The most defensible final claim is:

> Mixed static + adaptive recovery improves safety robustness under prefix-distribution shift, preserving original static-prefix robustness while reducing failures on adaptively mined prefixes. Trajectory analysis suggests this works by increasing refusal pressure at and after the prefix boundary, not simply by collapsing adaptive prefixes into the original static distribution.

## What Not To Overclaim

- Do not claim general safety robustness across all harmful prompt distributions unless the Aegis eval is run.
- Do not treat the 23-prompt gen3 set as a population-level benchmark; frame it as a paired hard-case mechanism set.
- Do not treat the 5-prefix gen4 remine ASR as stable. The keep rate is the real result there.
- Do not make teacher-forcing zigzag the main paper unless the utility story cleans up.

## File Map

- Figures: `logs/final_adaptive_ood_package/figures/`
- Aegis prepared files: `logs/final_adaptive_ood_package/aegis/`
- Aegis eval command: `{command_path}`
- Source report: `logs/adaptive_recovery/final_experiment_report.md`
"""
    path = OUT_DIR / "final_adaptive_ood_report.md"
    path.write_text(report)
    return path


def main() -> None:
    ensure_dirs()
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

    figs = {
        "static_baseline": plot_static_baseline(),
        "matched_static_vs_adaptive": plot_matched_static_vs_adaptive(),
        "recovery_static": plot_recovery_static(),
        "gen3_models": plot_gen3_models(),
        "gpt_static": plot_gpt_judge_static(),
        "trajectory_refusal_margin": plot_trajectory_refusal_margin(),
        "trajectory_signs": plot_trajectory_delta_signs(),
        "subspace": plot_subspace(),
    }
    aegis_metadata = prepare_aegis()
    figs["aegis_categories"] = Path(aegis_metadata["category_plot"])
    command_path = write_aegis_commands(aegis_metadata)
    report_path = write_report(figs, aegis_metadata, command_path)

    print(json.dumps({"report": str(report_path), "figures": {k: str(v) for k, v in figs.items()}, "aegis": aegis_metadata, "aegis_command": str(command_path)}, indent=2))


if __name__ == "__main__":
    main()
