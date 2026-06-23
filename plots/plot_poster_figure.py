"""
Poster figure: shallow vs deep alignment in refusal geometry.

Panel A — how much of each condition's hidden-state delta lies inside the
           8-dimensional refusal subspace (refusal subspace fraction).
Panel B — geometric overlap (mean cos² of principal angles) between the
           rank-8 delta subspaces of each pair of conditions.

Run: python plot_poster_figure.py
"""

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

SAVE_DIR = Path("logs/adaptive_prefixes/traj_multidim")
OUT_PDF = SAVE_DIR / "poster_figure.pdf"
OUT_PNG = SAVE_DIR / "poster_figure.png"

# ── data ─────────────────────────────────────────────────────────────────────
traj = pd.read_csv(SAVE_DIR / "trajectory_summary.csv")
traj = traj[(traj["layer"] == 28) & (traj["response_step"] == 0)].copy()

angles = pd.read_csv(SAVE_DIR / "subspace_angles.csv")
angles = angles[(angles["layer"] == 28) & (angles["response_step"] == 0)].copy()

ks = sorted(traj["k"].unique())

# ── style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)

COND = {
    "clean":    dict(color="#757575", ls="--", marker="s", lw=2.2, ms=8,  label="Clean (no prefix)"),
    "refusal":  dict(color="#1565c0", ls="-",  marker="o", lw=2.5, ms=9,  label="Refusal prefix"),
    "static":   dict(color="#e65100", ls="-",  marker="^", lw=2.2, ms=9,  label="Static harmful prefix"),
    "adaptive": dict(color="#c62828", ls="-",  marker="D", lw=2.2, ms=8,  label="Adaptive mined prefix"),
}

PAIR = {
    ("refusal", "static"):   dict(color="#6a1b9a", ls="--", marker="o", lw=2.2, ms=8, label="Refusal ↔ Static"),
    ("refusal", "adaptive"): dict(color="#1565c0", ls=":",  marker="D", lw=2.2, ms=8, label="Refusal ↔ Adaptive"),
    ("static",  "adaptive"): dict(color="#e65100", ls="-",  marker="^", lw=2.5, ms=9, label="Static ↔ Adaptive"),
}

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
fig.subplots_adjust(wspace=0.38)

# ─── Panel A: refusal subspace fraction ──────────────────────────────────────
ax = axes[0]
for cond in ["clean", "refusal", "static", "adaptive"]:
    sub = traj[traj["condition"] == cond].sort_values("k")
    s = COND[cond]
    ax.plot(
        sub["k"], sub["refusal_subspace_fraction_mean"],
        color=s["color"], ls=s["ls"], marker=s["marker"],
        lw=s["lw"], ms=s["ms"], label=s["label"], zorder=3,
    )
    ax.fill_between(
        sub["k"],
        (sub["refusal_subspace_fraction_mean"] - sub["refusal_subspace_fraction_std"]).clip(0),
        (sub["refusal_subspace_fraction_mean"] + sub["refusal_subspace_fraction_std"]).clip(1),
        alpha=0.10, color=s["color"],
    )

# Annotate the "ceiling" and the attack plateau
ax.annotate(
    "Attacks: ~65%",
    xy=(20, 0.671), xytext=(22, 0.50),
    fontsize=12, color="#c62828",
    arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.4),
)
ax.annotate(
    "Refusal: 78–95%",
    xy=(10, 0.889), xytext=(14, 0.975),
    fontsize=12, color="#1565c0",
    arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1.4),
)

ax.set_xlabel("Prefix length  k  (tokens)")
ax.set_ylabel("Fraction of hidden-state delta\ninside refusal subspace")
ax.set_title("A   Partial refusal activation under attack")
ax.set_xticks(ks)
ax.set_ylim(-0.04, 1.08)
ax.axhline(0, color="#bdbdbd", lw=0.8, ls="--")
ax.grid(True, alpha=0.25, axis="y")
ax.legend(loc="lower center", bbox_to_anchor=(0.3, -0.3),
          ncol=2, framealpha=0.9, handlelength=2.2)

# ─── Panel B: cos² subspace alignment ────────────────────────────────────────
ax = axes[1]
for (ca, cb), s in PAIR.items():
    sub = angles[
        (angles["condition_a"] == ca) & (angles["condition_b"] == cb)
    ].sort_values("k")
    ax.plot(
        sub["k"], sub["subspace_alignment_mean_cos2"],
        color=s["color"], ls=s["ls"], marker=s["marker"],
        lw=s["lw"], ms=s["ms"], label=s["label"], zorder=3,
    )

# Annotate key contrasts at k=10
ax.annotate(
    "Static ≈ Adaptive\n(attacks share geometry)",
    xy=(10, 0.608), xytext=(17, 0.66),
    fontsize=11.5, color="#e65100", ha="center",
    arrowprops=dict(arrowstyle="->", color="#e65100", lw=1.4),
)

ax.set_xlabel("Prefix length  k  (tokens)")
ax.set_ylabel("Subspace geometric alignment\n(mean  cos²θ  of principal angles)")
ax.set_title("B   Attacks share geometry with each other,\n       not with refusal")
ax.set_xticks(ks)
ax.set_ylim(0.1, 0.75)
ax.grid(True, alpha=0.25, axis="y")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3),
          ncol=3, framealpha=0.9, handlelength=2.2)

# ── shared caption strip ──────────────────────────────────────────────────────

for ax, letter in zip(axes, "AB"):
    ax.tick_params(length=4)

fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT_PDF} and {OUT_PNG}")
