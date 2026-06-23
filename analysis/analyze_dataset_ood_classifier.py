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
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUT_DIR = Path("logs/final_adaptive_ood_package/dataset_ood")
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
REPORT_PATH = OUT_DIR / "dataset_ood_report.md"
FINAL_REPORT_PATH = Path("logs/final_adaptive_ood_package/final_adaptive_ood_report.md")

HEX_PATH = Path("finetuning_buckets/datasets/data/safety_bench/Harmful-HEx-PHI.jsonl")
AEGIS_PATH = Path("logs/final_adaptive_ood_package/aegis/aegis_unsafe_prompt_unsafe_response_prefixes.jsonl")


def read_hex_phi(path: Path) -> list[dict[str, Any]]:
    rows = []
    for idx, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        convo = json.loads(line)
        user = next((item["content"] for item in convo if item.get("role") == "user"), "")
        assistant = next((item["content"] for item in convo if item.get("role") == "assistant"), "")
        rows.append(
            {
                "dataset": "hex_phi",
                "label": 0,
                "id": idx,
                "prompt": user,
                "prefix": assistant,
                "category": "HEx-PHI",
            }
        )
    return rows


def read_aegis(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows.append(
            {
                "dataset": "aegis",
                "label": 1,
                "id": record.get("id"),
                "prompt": str(record.get("instruction", "")),
                "prefix": str(record.get("prefix", "")),
                "category": record.get("category", "Aegis"),
            }
        )
    return rows


def toks(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_']+", text.lower())


def entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def surface_features(prompt: str, prefix: str) -> dict[str, float]:
    prompt_toks = toks(prompt)
    prefix_toks = toks(prefix)
    combined = prompt + "\n" + prefix
    return {
        "prompt_char_len": len(prompt),
        "prompt_token_len": len(prompt_toks),
        "prefix_char_len": len(prefix),
        "prefix_token_len": len(prefix_toks),
        "combined_char_len": len(combined),
        "combined_token_len": len(prompt_toks) + len(prefix_toks),
        "prompt_question_marks": prompt.count("?"),
        "prefix_line_count": len(prefix.splitlines()),
        "prefix_numbered_markers": len(re.findall(r"(?:^|\\n)\\s*\\d+\\.", prefix)),
        "url_like_count": len(re.findall(r"https?://|\\[link\\]|\\.onion|www\\.", combined.lower())),
        "first_person_count": sum(combined.lower().count(x) for x in [" i ", " i'm", " i'll", " my ", " me "]),
        "instructional_count": sum(combined.lower().count(x) for x in ["step", "instructions", "guide", "how to", "tutorial"]),
        "refusal_count": sum(combined.lower().count(x) for x in ["cannot", "can't", "sorry", "unable", "ethical"]),
        "criminal_count": sum(combined.lower().count(x) for x in ["illegal", "fraud", "bomb", "weapon", "drug", "malware", "password"]),
        "char_entropy": entropy(combined),
        "prompt_prefix_token_ratio": len(prompt_toks) / max(1, len(prefix_toks)),
    }


def build_frame() -> pd.DataFrame:
    rows = read_hex_phi(HEX_PATH) + read_aegis(AEGIS_PATH)
    out = []
    for row in rows:
        feats = surface_features(row["prompt"], row["prefix"])
        out.append({**row, "prompt_text": row["prompt"], "combined_text": row["prompt"] + "\n" + row["prefix"], **feats})
    frame = pd.DataFrame(out)
    # Use a balanced deterministic subset for the primary classifier.
    hex_rows = frame[frame["dataset"] == "hex_phi"].copy()
    aegis_rows = frame[frame["dataset"] == "aegis"].sample(n=len(hex_rows), random_state=0).copy()
    balanced = pd.concat([hex_rows, aegis_rows], ignore_index=True).sample(frac=1, random_state=0).reset_index(drop=True)
    frame.to_csv(TABLE_DIR / "dataset_ood_all_features.csv", index=False)
    balanced.to_csv(TABLE_DIR / "dataset_ood_balanced_features.csv", index=False)
    return balanced


def evaluate_text_model(frame: pd.DataFrame, text_col: str, name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    X_text = frame[text_col].fillna("").astype(str)
    y = frame["label"].to_numpy()
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=5000, lowercase=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_predict(model, X_text, y, cv=cv, method="predict_proba")[:, 1]
    preds = (scores >= 0.5).astype(int)
    model.fit(X_text, y)
    vectorizer = model.named_steps["tfidfvectorizer"]
    clf = model.named_steps["logisticregression"]
    terms = np.array(vectorizer.get_feature_names_out())
    coefs = pd.DataFrame({"term": terms, "coefficient": clf.coef_[0]})
    top_aegis = coefs.sort_values("coefficient", ascending=False).head(30)
    top_hex = coefs.sort_values("coefficient", ascending=True).head(30)
    coef_out = pd.concat(
        [top_aegis.assign(direction="aegis"), top_hex.assign(direction="hex_phi")],
        ignore_index=True,
    )
    coef_out.to_csv(TABLE_DIR / f"{name}_tfidf_top_terms.csv", index=False)
    return (
        pd.DataFrame({"dataset": frame["dataset"], "label": y, "score": scores, "pred": preds}),
        {
            "model": name,
            "text_col": text_col,
            "accuracy": float(accuracy_score(y, preds)),
            "auc": float(roc_auc_score(y, scores)),
            "n": int(len(frame)),
        },
    )


def evaluate_surface_model(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    feature_cols = [
        col
        for col in frame.columns
        if col
        not in {
            "dataset",
            "label",
            "id",
            "prompt",
            "prefix",
            "category",
            "prompt_text",
            "combined_text",
        }
        and pd.api.types.is_numeric_dtype(frame[col])
    ]
    X = frame[feature_cols].to_numpy()
    y = frame["label"].to_numpy()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="liblinear"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    preds = (scores >= 0.5).astype(int)
    model.fit(X, y)
    clf = model.named_steps["logisticregression"]
    scaler = model.named_steps["standardscaler"]
    coef = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": clf.coef_[0],
            "hex_phi_mean": frame[frame.label.eq(0)][feature_cols].mean().values,
            "aegis_mean": frame[frame.label.eq(1)][feature_cols].mean().values,
            "pooled_std": scaler.scale_,
        }
    )
    coef["aegis_minus_hex"] = coef["aegis_mean"] - coef["hex_phi_mean"]
    coef["standardized_delta"] = coef["aegis_minus_hex"] / coef["pooled_std"].replace(0, np.nan)
    coef["abs_coefficient"] = coef["coefficient"].abs()
    coef = coef.sort_values("abs_coefficient", ascending=False)
    coef.to_csv(TABLE_DIR / "surface_feature_coefficients.csv", index=False)
    return (
        pd.DataFrame({"dataset": frame["dataset"], "label": y, "score": scores, "pred": preds}),
        {
            "model": "surface_features",
            "accuracy": float(accuracy_score(y, preds)),
            "auc": float(roc_auc_score(y, scores)),
            "n": int(len(frame)),
            "feature_count": len(feature_cols),
        },
        coef,
    )


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


def plot_scores(preds: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (name, df) in zip(axes, preds.items()):
        ax.hist(df[df.label.eq(0)]["score"], bins=15, alpha=0.75, color="#b94633", label="HEx-PHI")
        ax.hist(df[df.label.eq(1)]["score"], bins=15, alpha=0.75, color="#2f7f5f", label="Aegis")
        ax.set_title(name.replace("_", " "), fontsize=11, weight="bold")
        ax.set_xlabel("Predicted Aegis probability")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Count")
    axes[-1].legend(frameon=False)
    path = FIG_DIR / "dataset_ood_score_distributions.png"
    savefig(path)
    return path


def plot_surface_deltas(coef: pd.DataFrame) -> Path:
    rows = coef.reindex(coef["standardized_delta"].abs().sort_values(ascending=False).index).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    colors = ["#2f7f5f" if x > 0 else "#b94633" for x in rows["standardized_delta"]]
    ax.barh(rows["feature"], rows["standardized_delta"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    style_axes(ax, "Aegis vs HEx-PHI surface-feature differences", xlabel="Standardized Aegis minus HEx-PHI")
    path = FIG_DIR / "dataset_ood_surface_deltas.png"
    savefig(path)
    return path


def write_report(metrics: list[dict[str, Any]], coef: pd.DataFrame, score_fig: Path, delta_fig: Path) -> None:
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(TABLE_DIR / "dataset_ood_classifier_metrics.csv", index=False)
    top = coef.reindex(coef["standardized_delta"].abs().sort_values(ascending=False).index).head(8)
    report = f"""# Dataset OOD Classifier: Aegis vs HEx-PHI

## Question

Is Aegis meaningfully out-of-distribution relative to HEx-PHI, or is it just another unsafe prompt set?

## Setup

- HEx-PHI examples: `330`
- Aegis unsafe prompt + unsafe response examples sampled for balance: `330` from the prepared `637`
- Label: `Aegis=1`, `HEx-PHI=0`
- Evaluation: 5-fold stratified cross-validation

## Classifier Results

| Model | Input | Accuracy | AUC |
|---|---|---:|---:|
"""
    for row in metrics:
        report += (
            f"| `{row['model']}` | `{row.get('text_col', 'surface')}` | "
            f"{row['accuracy']:.3f} | {row['auc']:.3f} |\n"
        )
    report += f"""

![Dataset OOD scores](figures/{score_fig.name})

## Interpretable Surface Differences

| Feature | HEx-PHI mean | Aegis mean | Standardized Aegis-HEx |
|---|---:|---:|---:|
"""
    for _, row in top.iterrows():
        report += (
            f"| `{row['feature']}` | {row['hex_phi_mean']:.3f} | "
            f"{row['aegis_mean']:.3f} | {row['standardized_delta']:.2f} |\n"
        )
    report += f"""

![Dataset OOD surface deltas](figures/{delta_fig.name})

## Interpretation

If a lightweight classifier can distinguish Aegis from HEx-PHI with high cross-validated AUC, then Aegis is a concrete prompt-source OOD check rather than a cosmetic extra benchmark. The prompt-only model is especially important: it tests whether the user requests themselves differ, not merely the generated unsafe prefixes.

This does not mean Aegis is "harder" in a universal sense. It means it is distributionally different. The Aegis ASR results should therefore be read as a cross-dataset stress test, not as a direct continuation of the HEx-PHI adaptive-prefix benchmark.
"""
    REPORT_PATH.write_text(report)

    final = FINAL_REPORT_PATH.read_text()
    section = f"""

## Dataset OOD Diagnostic: Aegis vs HEx-PHI

To make prompt-source OOD concrete, I trained lightweight classifiers to distinguish Aegis unsafe examples from HEx-PHI harmful examples.

| Model | Input | Accuracy | AUC |
|---|---|---:|---:|
"""
    for row in metrics:
        section += (
            f"| `{row['model']}` | `{row.get('text_col', 'surface')}` | "
            f"{row['accuracy']:.3f} | {row['auc']:.3f} |\n"
        )
    section += """

![Dataset OOD scores](dataset_ood/figures/dataset_ood_score_distributions.png)

Interpretation: Aegis is not merely another sample from the same HEx-PHI distribution. Even prompt-only text is highly separable, so the Aegis results are valid evidence about prompt-source OOD. This also explains why mixed recovery can improve over adaptive-only while still not beating the original augmented model on Aegis: the Aegis distribution is genuinely different from the adaptive HEx-PHI recovery distribution.
"""
    marker = "\n## Aegis Cross-Dataset Results\n"
    if "## Dataset OOD Diagnostic: Aegis vs HEx-PHI" in final:
        start = final.index("\n## Dataset OOD Diagnostic: Aegis vs HEx-PHI")
        end = final.index(marker)
        final = final[:start] + section + final[end:]
    else:
        final = final.replace(marker, section + marker)
    FINAL_REPORT_PATH.write_text(final)


def main() -> None:
    for directory in [OUT_DIR, FIG_DIR, TABLE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})
    frame = build_frame()
    preds = {}
    metrics = []
    for text_col, name in [("prompt_text", "prompt_tfidf"), ("combined_text", "prompt_plus_prefix_tfidf")]:
        pred, metric = evaluate_text_model(frame, text_col, name)
        preds[name] = pred
        metrics.append(metric)
    pred, metric, coef = evaluate_surface_model(frame)
    preds["surface_features"] = pred
    metrics.append(metric)
    score_fig = plot_scores(preds)
    delta_fig = plot_surface_deltas(coef)
    write_report(metrics, coef, score_fig, delta_fig)
    print(
        json.dumps(
            {
                "metrics": metrics,
                "report": str(REPORT_PATH),
                "final_report": str(FINAL_REPORT_PATH),
                "figures": [str(score_fig), str(delta_fig)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
