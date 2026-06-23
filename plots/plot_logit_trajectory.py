"""
Refusal-zone fraction vs response step (steps 0-40, static & adaptive, k=5 & k=40).
Input:  logs/adaptive_prefixes/logit_traj/logit_trajectory.csv
Output: logs/adaptive_prefixes/logit_traj/logit_trajectory_agg.{pdf,png}
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

SAVE_DIR = Path("logs/adaptive_prefixes/logit_traj")
IN_CSV   = SAVE_DIR / "logit_trajectory.csv"

df = pd.read_csv(IN_CSV)
KS         = [5, 40]
CONDITIONS = ["static", "adaptive"]
STEPS      = sorted(df["response_step"].unique())

C_STATIC   = "#e65100"
C_ADAPTIVE = "#c62828"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(9, 5))

style = {
    ("static",   5):  dict(color=C_STATIC,   ls="--", marker="^", lw=2.2, ms=7, label="Static $k{=}5$"),
    ("static",  40):  dict(color=C_STATIC,   ls="-",  marker="^", lw=2.6, ms=8, label="Static $k{=}40$"),
    ("adaptive", 5):  dict(color=C_ADAPTIVE, ls="--", marker="D", lw=2.2, ms=7, label="Adaptive $k{=}5$"),
    ("adaptive",40):  dict(color=C_ADAPTIVE, ls="-",  marker="D", lw=2.6, ms=8, label="Adaptive $k{=}40$"),
}

for (cond, k), sty in style.items():
    sub = df[(df["condition"] == cond) & (df["k"] == k)]
    frac = (
        sub.groupby("response_step")["refusal_margin"]
        .apply(lambda v: (v > 0).mean())
        .reset_index()
    )
    frac.columns = ["step", "frac"]
    frac = frac.sort_values("step")
    ax.plot(frac["step"], frac["frac"], **sty)

ax.set_xlabel("Response step")
ax.set_ylabel("Refusal-zone fraction  (margin > 0)")
ax.set_title("Refusal-zone fraction across response steps")
ax.set_xticks(range(0, max(STEPS) + 1, 5))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.grid(True, alpha=0.22, axis="y")
ax.legend(framealpha=0.9, handlelength=2.2)

fig.tight_layout()

out_pdf = SAVE_DIR / "logit_trajectory_agg.pdf"
out_png = SAVE_DIR / "logit_trajectory_agg.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_pdf} and {out_png}")
