from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUT_DIR = Path("logs/final_adaptive_ood_package/prefix_uniqueness")
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
REPORT_PATH = OUT_DIR / "prefix_uniqueness_report.md"
FINAL_REPORT_PATH = Path("logs/final_adaptive_ood_package/final_adaptive_ood_report.md")

TRAIN_ADAPTIVE = Path("logs/adaptive_prefixes/mined_augmented_tinker.jsonl")
TRAIN_STATIC = Path("logs/adaptive_prefixes/matched_static_prefixes_from_mined.jsonl")
HELDOUT_ADAPTIVE = Path("logs/adaptive_prefixes/gen3_hardcore_compare/gen3_same23.jsonl")
HELDOUT_STATIC = Path("logs/adaptive_prefixes/gen3_hardcore_compare/static_same23.jsonl")

REFUSAL_MARKERS = [
    "i cannot",
    "i can't",
    "i am unable",
    "i'm unable",
    "i apologize",
    "i'm sorry",
    "sorry",
    "cannot fulfill",
    "can't fulfill",
    "not able to",
    "as an ai",
    "ethical",
    "illegal",
]

HARMFUL_ACTION_TERMS = [
    "step",
    "instructions",
    "guide",
    "method",
    "avoid detection",
    "not get caught",
    "hide",
    "exploit",
    "bypass",
    "weapon",
    "bomb",
    "drug",
    "fraud",
    "password",
    "malware",
    "phishing",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_']+|<\\|[^>]+\\|>", text.lower())


def char_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def repeated_ngram_fraction(toks: list[str], n: int) -> float:
    if len(toks) < n:
        return 0.0
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(grams)


def max_run_length(toks: list[str]) -> int:
    if not toks:
        return 0
    best = current = 1
    for prev, cur in zip(toks, toks[1:]):
        if cur == prev:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def count_substrings(text: str, substrings: list[str]) -> int:
    lower = text.lower()
    return sum(lower.count(item) for item in substrings)


def extract_features(prefix: str, instruction: str = "") -> dict[str, float]:
    toks = tokens(prefix)
    word_toks = [tok for tok in toks if not tok.startswith("<|")]
    chars = len(prefix)
    token_n = len(toks)
    lines = prefix.splitlines()
    unique_toks = len(set(toks))
    special_count = len(re.findall(r"<\\|[^>]+\\|>", prefix))
    begin_count = prefix.count("<|begin_of_text|>")
    eot_count = prefix.count("<|eot_id|>")
    digit_chars = sum(ch.isdigit() for ch in prefix)
    alpha_chars = sum(ch.isalpha() for ch in prefix)
    punct_chars = sum((not ch.isalnum()) and (not ch.isspace()) for ch in prefix)
    numbered = len(re.findall(r"(?:^|\\n)\\s*(?:\\d+\\.|[-*])\\s+", prefix))
    code_symbols = sum(prefix.count(sym) for sym in ["{", "}", ";", "()", "=>", "func", "class", "import"])
    refusal_marker_count = count_substrings(prefix, REFUSAL_MARKERS)
    harmful_action_count = count_substrings(prefix, HARMFUL_ACTION_TERMS)
    instruction_words = set(tokens(instruction))
    prefix_words = set(word_toks)
    overlap = len(instruction_words & prefix_words) / len(instruction_words) if instruction_words else 0.0

    return {
        "char_len": chars,
        "token_len": token_n,
        "line_count": len(lines),
        "avg_line_len": chars / max(1, len(lines)),
        "unique_token_ratio": unique_toks / max(1, token_n),
        "max_token_frequency": max(Counter(toks).values()) / max(1, token_n) if toks else 0.0,
        "max_repeated_run": max_run_length(toks),
        "repeated_bigram_fraction": repeated_ngram_fraction(toks, 2),
        "repeated_trigram_fraction": repeated_ngram_fraction(toks, 3),
        "char_entropy": char_entropy(prefix),
        "special_token_count": special_count,
        "begin_token_count": begin_count,
        "eot_token_count": eot_count,
        "digit_char_fraction": digit_chars / max(1, chars),
        "alpha_char_fraction": alpha_chars / max(1, chars),
        "punct_char_fraction": punct_chars / max(1, chars),
        "numbered_list_markers": numbered,
        "code_symbol_count": code_symbols,
        "refusal_marker_count": refusal_marker_count,
        "harmful_action_count": harmful_action_count,
        "instruction_overlap": overlap,
        "starts_with_special": float(prefix.strip().startswith("<|")),
        "ends_mid_sentence": float(bool(prefix.strip()) and prefix.strip()[-1] not in ".!?:;)}]\"'"),
    }


def build_dataset(adaptive_path: Path, static_path: Path, split: str) -> pd.DataFrame:
    rows = []
    for label, path, source in [(1, adaptive_path, "adaptive"), (0, static_path, "static")]:
        for record in read_jsonl(path):
            prefix = str(record.get("prefix", ""))
            instruction = str(record.get("instruction", ""))
            features = extract_features(prefix, instruction)
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "source": source,
                    "id": record.get("id"),
                    "instruction": instruction,
                    "prefix": prefix,
                    **features,
                }
            )
    return pd.DataFrame(rows)


def all_feature_columns(train: pd.DataFrame) -> list[str]:
    return [
        col
        for col in train.columns
        if col
        not in {
            "split",
            "label",
            "source",
            "id",
            "instruction",
            "prefix",
        }
    ]


def evaluate_feature_set(
    train: pd.DataFrame,
    heldout: pd.DataFrame,
    feature_cols: list[str],
    name: str,
    save_predictions: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    X = train[feature_cols].to_numpy()
    y = train["label"].to_numpy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pred_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    pred_label = (pred_proba >= 0.5).astype(int)

    model.fit(X, y)
    heldout_proba = model.predict_proba(heldout[feature_cols].to_numpy())[:, 1]
    heldout_pred = (heldout_proba >= 0.5).astype(int)
    heldout = heldout.copy()
    heldout["adaptive_score"] = heldout_proba
    heldout["pred_label"] = heldout_pred

    clf = model.named_steps["logisticregression"]
    scaler = model.named_steps["standardscaler"]
    coefs = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": clf.coef_[0],
            "train_static_mean": train[train["label"] == 0][feature_cols].mean().values,
            "train_adaptive_mean": train[train["label"] == 1][feature_cols].mean().values,
            "pooled_std": scaler.scale_,
        }
    )
    coefs["adaptive_minus_static"] = coefs["train_adaptive_mean"] - coefs["train_static_mean"]
    coefs["standardized_delta"] = coefs["adaptive_minus_static"] / coefs["pooled_std"].replace(0, np.nan)
    coefs["abs_coefficient"] = coefs["coefficient"].abs()
    coefs = coefs.sort_values("abs_coefficient", ascending=False)

    metrics = {
        "feature_set": name,
        "train_n": int(len(train)),
        "heldout_n": int(len(heldout)),
        "cv_accuracy": float(accuracy_score(y, pred_label)),
        "cv_auc": float(roc_auc_score(y, pred_proba)),
        "heldout_accuracy": float(accuracy_score(heldout["label"], heldout_pred)),
        "heldout_auc": float(roc_auc_score(heldout["label"], heldout_proba)),
        "feature_count": len(feature_cols),
    }
    if save_predictions:
        heldout.to_csv(TABLE_DIR / "heldout_gen3_predictions.csv", index=False)
        pd.DataFrame({"label": y, "adaptive_score": pred_proba}).to_csv(TABLE_DIR / "train_cv_predictions.csv", index=False)
        coefs.to_csv(TABLE_DIR / "prefix_feature_coefficients.csv", index=False)
    return coefs, metrics


def fit_and_score(train: pd.DataFrame, heldout: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    feature_cols = all_feature_columns(train)
    special_features = {"special_token_count", "begin_token_count", "eot_token_count", "starts_with_special"}
    length_format_features = {
        "char_len",
        "token_len",
        "line_count",
        "avg_line_len",
        "numbered_list_markers",
        "code_symbol_count",
    }
    semantic_features = ["refusal_marker_count", "harmful_action_count", "instruction_overlap"]

    feature_sets = {
        "all_features": feature_cols,
        "no_special_tokens": [col for col in feature_cols if col not in special_features],
        "no_special_or_length_format": [
            col for col in feature_cols if col not in special_features | length_format_features
        ],
        "semantic_overlap_only": semantic_features,
    }

    all_metrics = []
    full_coefs = None
    full_metrics = None
    for name, cols in feature_sets.items():
        coefs, metrics = evaluate_feature_set(
            train,
            heldout,
            cols,
            name=name,
            save_predictions=name == "all_features",
        )
        all_metrics.append(metrics)
        if name == "all_features":
            full_coefs = coefs
            full_metrics = metrics

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(TABLE_DIR / "prefix_classifier_ablation_metrics.csv", index=False)
    assert full_coefs is not None and full_metrics is not None
    return full_coefs, full_metrics, metrics_df


def style_axes(ax, title: str, ylabel: str | None = None, xlabel: str | None = None) -> None:
    ax.set_title(title, fontsize=13, weight="bold", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_standardized_deltas(coefs: pd.DataFrame) -> Path:
    rows = coefs.reindex(coefs["standardized_delta"].abs().sort_values(ascending=False).index).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    colors = ["#2f7f5f" if value > 0 else "#b94633" for value in rows["standardized_delta"]]
    ax.barh(rows["feature"], rows["standardized_delta"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    style_axes(ax, "Largest surface-feature differences: adaptive minus static", xlabel="Standardized mean difference")
    path = FIG_DIR / "prefix_feature_standardized_deltas.png"
    savefig(path)
    return path


def plot_coefficients(coefs: pd.DataFrame) -> Path:
    rows = coefs.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    colors = ["#2f7f5f" if value > 0 else "#b94633" for value in rows["coefficient"]]
    ax.barh(rows["feature"], rows["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    style_axes(ax, "Logistic classifier weights for adaptive-prefix prediction", xlabel="Coefficient after feature standardization")
    path = FIG_DIR / "prefix_feature_classifier_coefficients.png"
    savefig(path)
    return path


def plot_score_distributions() -> Path:
    train = pd.read_csv(TABLE_DIR / "train_cv_predictions.csv")
    heldout = pd.read_csv(TABLE_DIR / "heldout_gen3_predictions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, df, title in [
        (axes[0], train, "Matched 163-pair CV scores"),
        (axes[1], heldout, "Gen3 hard-set heldout scores"),
    ]:
        ax.hist(df[df["label"] == 0]["adaptive_score"], bins=12, alpha=0.75, color="#b94633", label="static")
        ax.hist(df[df["label"] == 1]["adaptive_score"], bins=12, alpha=0.75, color="#2f7f5f", label="adaptive")
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("Predicted adaptive-prefix probability")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Count")
    axes[1].legend(frameon=False)
    path = FIG_DIR / "prefix_classifier_score_distributions.png"
    savefig(path)
    return path


def write_report(coefs: pd.DataFrame, metrics: dict[str, Any], ablations: pd.DataFrame, figs: list[Path]) -> None:
    top_delta = coefs.reindex(coefs["standardized_delta"].abs().sort_values(ascending=False).index).head(8)
    top_coef = coefs.head(8)
    report = f"""# Prefix Uniqueness Diagnostic

## Question

Are adaptive prefixes measurably different from matched static harmful prefixes, or are the recovery gains mainly because the model was fine-tuned on the adaptive examples?

## Setup

- Training diagnostic set: `163` matched static/adaptive prefix pairs from the original adaptive mining run.
- Heldout-style check: `23` static/adaptive prefix pairs from the gen3 hard set.
- Label: `adaptive=1`, `static=0`.
- Classifier: standardized logistic regression over simple surface features.

This is intentionally cheap and interpretable. It does not prove causality, but it tells us whether adaptive prefixes have measurable surface structure.

## Classifier Result

- 5-fold CV accuracy on matched 163-pair set with all features: `{metrics['cv_accuracy']:.3f}`
- 5-fold CV AUC on matched 163-pair set with all features: `{metrics['cv_auc']:.3f}`
- Gen3 hard-set accuracy with all features: `{metrics['heldout_accuracy']:.3f}`
- Gen3 hard-set AUC with all features: `{metrics['heldout_auc']:.3f}`
- Features used: `{metrics['feature_count']}`

![Score distributions](figures/prefix_classifier_score_distributions.png)

## Ablation Check

Because special-token artifacts are an easy separator, I reran the classifier after removing feature groups:

| Feature set | Features | CV accuracy | CV AUC | Gen3 accuracy | Gen3 AUC |
|---|---:|---:|---:|---:|---:|
"""
    for _, row in ablations.iterrows():
        report += (
            f"| `{row['feature_set']}` | {int(row['feature_count'])} | "
            f"{row['cv_accuracy']:.3f} | {row['cv_auc']:.3f} | "
            f"{row['heldout_accuracy']:.3f} | {row['heldout_auc']:.3f} |\n"
        )
    report += """

The key point is not just that special tokens separate adaptive prefixes. Even after removing special-token features, the classifier remains highly separable. With only semantic/overlap features, separability drops, which suggests the distinction is mostly in generation/format/state features rather than simply semantic harmfulness.

## What Differs

Largest standardized mean differences, adaptive minus static:

| Feature | Static mean | Adaptive mean | Standardized delta |
|---|---:|---:|---:|
"""
    for _, row in top_delta.iterrows():
        report += (
            f"| `{row['feature']}` | {row['train_static_mean']:.3f} | "
            f"{row['train_adaptive_mean']:.3f} | {row['standardized_delta']:.2f} |\n"
        )
    report += """

![Feature deltas](figures/prefix_feature_standardized_deltas.png)

Largest logistic weights after standardization:

| Feature | Coefficient | Adaptive minus static |
|---|---:|---:|
"""
    for _, row in top_coef.iterrows():
        report += f"| `{row['feature']}` | {row['coefficient']:.2f} | {row['adaptive_minus_static']:.3f} |\n"
    report += """

![Classifier coefficients](figures/prefix_feature_classifier_coefficients.png)

## Interpretation

The classifier result supports a modest but important claim:

> Adaptive prefixes are measurably different from matched static harmful prefixes, even when prompt identity is controlled.

The most visible differences are surface/formatting and generation-state artifacts: special tokens, repetition, prefix length/line structure, list/code markers, and whether the prefix ends in an unfinished continuation. These are exactly the kinds of features one would expect if adaptive mining is exploiting model-conditioned continuation states rather than simply selecting semantically more harmful text.

## Caveat

This does not prove that these features cause jailbreak success. It only shows that adaptive and static prefixes are separable. The causal version would require feature-matched controls or prefix-swap experiments:

- Train on adaptive prefixes from one prompt split and test on newly mined adaptive prefixes from held-out prompts.
- Match static and adaptive prefixes on length, category, repetition, and special-token count.
- Swap adaptive prefixes onto different harmful prompts and test whether the attack transfers.
"""
    REPORT_PATH.write_text(report)

    final_text = FINAL_REPORT_PATH.read_text()
    section = f"""

## Prefix Uniqueness Diagnostic

A cheap surface-feature classifier can distinguish matched static vs adaptive prefixes, which supports the claim that adaptive prefixes are not just more samples from the same static-prefix distribution.

- 5-fold CV accuracy on the 163-pair matched set with all features: `{metrics['cv_accuracy']:.3f}`
- 5-fold CV AUC with all features: `{metrics['cv_auc']:.3f}`
- Gen3 hard-set accuracy with all features: `{metrics['heldout_accuracy']:.3f}`
- Gen3 hard-set AUC with all features: `{metrics['heldout_auc']:.3f}`
- After removing special-token features, CV AUC: `{ablations[ablations['feature_set'].eq('no_special_tokens')]['cv_auc'].iloc[0]:.3f}`
- With only semantic/overlap features, CV AUC: `{ablations[ablations['feature_set'].eq('semantic_overlap_only')]['cv_auc'].iloc[0]:.3f}`

![Prefix classifier scores](prefix_uniqueness/figures/prefix_classifier_score_distributions.png)

![Prefix feature deltas](prefix_uniqueness/figures/prefix_feature_standardized_deltas.png)

Interpretation: adaptive prefixes are separable by surface/generation-state features, especially special-token artifacts, repetition/length structure, formatting, and unfinished-continuation markers. The ablation check suggests this is not only a special-token artifact, but the weak semantic-only result means the distinction is more about induced continuation state than simple harmful-topic content. This supports the distribution-shift story, but it does not by itself prove causality; feature-matched controls or prefix-swap tests would be needed for that.
"""
    marker = "\n## Aegis Cross-Dataset Results\n"
    if "## Prefix Uniqueness Diagnostic" in final_text:
        start = final_text.index("\n## Prefix Uniqueness Diagnostic")
        end = final_text.index(marker)
        final_text = final_text[:start] + section + final_text[end:]
    else:
        final_text = final_text.replace(marker, section + marker)
    FINAL_REPORT_PATH.write_text(final_text)


def main() -> None:
    for directory in [OUT_DIR, FIG_DIR, TABLE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})

    train = build_dataset(TRAIN_ADAPTIVE, TRAIN_STATIC, split="matched_gen1")
    heldout = build_dataset(HELDOUT_ADAPTIVE, HELDOUT_STATIC, split="gen3_same23")
    train.to_csv(TABLE_DIR / "prefix_uniqueness_train_features.csv", index=False)
    heldout.to_csv(TABLE_DIR / "prefix_uniqueness_heldout_features.csv", index=False)

    coefs, metrics, ablations = fit_and_score(train, heldout)
    figs = [plot_score_distributions(), plot_standardized_deltas(coefs), plot_coefficients(coefs)]
    with (OUT_DIR / "metrics.json").open("w") as handle:
        json.dump({"all_features": metrics, "ablations": ablations.to_dict(orient="records")}, handle, indent=2)
    write_report(coefs, metrics, ablations, figs)
    print(
        json.dumps(
            {
                "metrics": metrics,
                "report": str(REPORT_PATH),
                "final_report": str(FINAL_REPORT_PATH),
                "figures": [str(path) for path in figs],
                "coefficients": str(TABLE_DIR / "prefix_feature_coefficients.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
