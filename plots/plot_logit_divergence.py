"""
Logit distribution divergence at large k: static vs adaptive prefixes.

Panel A — KDE of per-example refusal margin at k=5 and k=40 for static and
           adaptive, with the compliance zone (margin < 0) shaded.
Panel B — Compliance-zone fraction (fraction of examples with margin < 0)
           across k for static and adaptive.
Panel C — KL divergence to refusal distribution across k, same conditions.

All measurements are at the final layer (28), response step 0 (first free
token after the forced prefix), on the Llama-3.2-3B-Augmented model.
"""

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

SAVE_DIR = Path("logs/adaptive_prefixes/traj_multidim")
KS = [5, 10, 20, 40]
LAYER, STEP = 28, 0

# ── load data ─────────────────────────────────────────────────────────────────
summary = pd.read_csv(SAVE_DIR / "trajectory_summary.csv")
summary = summary[(summary["layer"] == LAYER) & (summary["response_step"] == STEP)]

per_example = pd.concat(
    [
        pd.read_csv(SAVE_DIR / f"trajectory_per_example_k{k}.csv").assign(k=k)
        for k in KS
    ],
    ignore_index=True,
)
per_example = per_example[
    (per_example["layer"] == LAYER)
    & (per_example["response_step"] == STEP)
    & (per_example["condition"].isin(["static", "adaptive", "clean", "refusal"]))
].copy()

# ── style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

C_STATIC   = "#e65100"
C_ADAPTIVE = "#c62828"
C_REFUSAL  = "#1565c0"
C_CLEAN    = "#757575"

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5.2))
gs = fig.add_gridspec(1, 3, wspace=0.40)
ax_kde = fig.add_subplot(gs[0, 0])
ax_frac = fig.add_subplot(gs[0, 1])
ax_kl   = fig.add_subplot(gs[0, 2])

# ─── Panel A: KDE of refusal margin at k=5 and k=40 ─────────────────────────
margin_range = np.linspace(-18, 18, 600)

style_kde = {
    ("static",   5):  dict(color=C_STATIC,   lw=1.8, ls="--", label="Static  k=5"),
    ("static",  40):  dict(color=C_STATIC,   lw=2.8, ls="-",  label="Static  k=40"),
    ("adaptive", 5):  dict(color=C_ADAPTIVE, lw=1.8, ls="--", label="Adaptive k=5"),
    ("adaptive",40):  dict(color=C_ADAPTIVE, lw=2.8, ls="-",  label="Adaptive k=40"),
}

kde_data = {}
for (cond, k), sty in style_kde.items():
    vals = per_example.loc[
        (per_example["condition"] == cond) & (per_example["k"] == k),
        "refusal_margin",
    ].dropna().values
    if len(vals) < 5:
        continue
    kde = gaussian_kde(vals, bw_method=0.35)
    density = kde(margin_range)
    ax_kde.plot(margin_range, density, **sty)
    kde_data[(cond, k)] = (vals, density)
    if k == 40:
        mask = margin_range < 0
        ax_kde.fill_between(margin_range[mask], density[mask], alpha=0.13, color=sty["color"])

ax_kde.axvline(0, color="#424242", lw=1.2, ls=":", label="Compliance boundary")
ax_kde.set_xlabel("Refusal margin  (log P(refusal) − log P(comply))")
ax_kde.set_ylabel("Density")
ax_kde.set_title("A   First-token logit distribution\n    (k = 5 vs 40)")
ax_kde.set_xlim(-17, 14)
ax_kde.axvspan(-17, 0, alpha=0.04, color="#424242")

# annotate compliance fractions AFTER ylim is set by data
ymax = max(d.max() for d in [kde_data[(c,k)][1] for c,k in kde_data])
ax_kde.set_ylim(bottom=0, top=ymax * 1.25)

for cond, color, xann, xarrow in [
    ("static",   C_STATIC,   -12.5, -6.5),
    ("adaptive", C_ADAPTIVE,  -9.5, -3.5),
]:
    vals, density = kde_data[(cond, 40)]
    frac = float((vals < 0).mean())
    mask = margin_range < 0
    ax_kde.annotate(
        f"{frac:.0%}  ({cond} k=40)",
        xy=(xarrow, gaussian_kde(vals, bw_method=0.35)(np.array([xarrow]))[0]),
        xytext=(xann, ymax * (0.90 if cond == "static" else 0.74)),
        fontsize=11, color=color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=color, lw=1.3),
    )

ax_kde.text(-8.5, ymax * 1.15, "← compliance biased", fontsize=10, color="#616161", ha="center")
ax_kde.text( 6.5, ymax * 1.15, "refusal biased →",    fontsize=10, color="#616161", ha="center")
ax_kde.legend(loc="upper right", framealpha=0.85, handlelength=2)

# ─── Panel B: compliance-zone fraction vs k ───────────────────────────────────
frac_rows = []
for cond in ["static", "adaptive"]:
    for k in KS:
        vals = per_example.loc[
            (per_example["condition"] == cond) & (per_example["k"] == k),
            "refusal_margin",
        ].dropna()
        frac_rows.append({"condition": cond, "k": k, "compliance_frac": (vals < 0).mean()})
frac_df = pd.DataFrame(frac_rows)

for cond, color, marker, label in [
    ("static",   C_STATIC,   "^", "Static harmful prefix"),
    ("adaptive", C_ADAPTIVE, "D", "Adaptive mined prefix"),
]:
    sub = frac_df[frac_df["condition"] == cond].sort_values("k")
    ax_frac.plot(
        sub["k"], sub["compliance_frac"],
        color=color, marker=marker, lw=2.4, ms=9, label=label,
    )

# Annotate the divergence at k=40
s40 = frac_df[(frac_df["condition"]=="static")  & (frac_df["k"]==40)]["compliance_frac"].values[0]
a40 = frac_df[(frac_df["condition"]=="adaptive") & (frac_df["k"]==40)]["compliance_frac"].values[0]
ax_frac.annotate(
    f"Δ = {s40-a40:.1%}\nat k=40",
    xy=(40, (s40 + a40) / 2),
    xytext=(28, (s40 + a40) / 2 + 0.03),
    fontsize=11, color="#424242",
    arrowprops=dict(arrowstyle="-[,widthB=1.5", color="#424242", lw=1.2),
)

ax_frac.set_xlabel("Prefix length  k  (tokens)")
ax_frac.set_ylabel("Fraction of examples\nwith compliance-biased first token")
ax_frac.set_title("B   Compliance-zone fraction\n    grows faster for static at large k")
ax_frac.set_xticks(KS)
ax_frac.set_ylim(0.10, 0.42)
ax_frac.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
ax_frac.grid(True, alpha=0.25, axis="y")
ax_frac.legend(framealpha=0.9)

# ─── Panel C: KL to refusal vs k ─────────────────────────────────────────────
kl_style = {
    "static":   dict(color=C_STATIC,   marker="^", lw=2.4, ms=9, label="Static"),
    "adaptive": dict(color=C_ADAPTIVE, marker="D", lw=2.4, ms=9, label="Adaptive"),
    "clean":    dict(color=C_CLEAN,    marker="s", lw=1.6, ms=7, ls="--", label="Clean"),
}
for cond, sty in kl_style.items():
    sub = summary[summary["condition"] == cond].sort_values("k")
    ax_kl.plot(sub["k"], sub["kl_to_refusal_mean"], **sty)
    ax_kl.fill_between(
        sub["k"],
        sub["kl_to_refusal_mean"] - sub["kl_to_refusal_std"],
        sub["kl_to_refusal_mean"] + sub["kl_to_refusal_std"],
        alpha=0.10, color=sty["color"],
    )

ax_kl.set_xlabel("Prefix length  k  (tokens)")
ax_kl.set_ylabel("KL divergence to refusal distribution\n(nats)")
ax_kl.set_title("C   Full-vocabulary divergence from\n    refusal output distribution")
ax_kl.set_xticks(KS)
ax_kl.grid(True, alpha=0.25, axis="y")
ax_kl.legend(framealpha=0.9)

# ── shared caption ────────────────────────────────────────────────────────────
fig.text(
    0.5, -0.04,
    "Layer 28, response step 0 (first free token logits after forced prefix)  ·  "
    "Llama-3.2-3B-Augmented  ·  n = 163 matched examples",
    ha="center", fontsize=11, color="#424242", style="italic",
)

fig.savefig(SAVE_DIR / "logit_divergence.pdf", bbox_inches="tight")
fig.savefig(SAVE_DIR / "logit_divergence.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved logit_divergence.{pdf,png}")
