"""Redraw Figures 2 and 3 so that they match the manuscript captions exactly.

  Fig 2  World Athletics series drawn as a DASHED line with TRIANGLE markers
         (caption: "dashed lines with triangles represent World Athletics (WA)
         reference values"). Group B is a plain dotted line with no marker
         (caption: "dotted lines near 1.0"), so the triangle is unambiguous.
  Fig 2  Group A percentages along the bottom of each panel (caption:
         "Percentages at the bottom indicate the proportion of Group A
         athletes at each cutoff age").
  Fig 3  World Athletics series likewise dashed with triangle markers.

WA reference series = the 500-repeat cutoff-curve sweep of the World Athletics
data (cutoff ages 16-26, GradientBoosting, random 5-fold CV). The same numbers
are carried in the `wa_r2` column of results/tables/composition_decomposition.csv
and in WA_REF inside src/diagnostic_r2_inflation.py; WA_CURVE below is kept
explicit so the figure can be redrawn from the shipped table alone.

Input:   results/tables/composition_decomposition.csv
Output:  results/figures/Figure{2,3}_redraw.{png,pdf}
Run:     python redraw_figures_2_3.py
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV = HERE / "results" / "tables" / "composition_decomposition.csv"
OUT = HERE / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

if not CSV.exists():
    raise SystemExit(f"missing input table: {CSV}")

with open(CSV, newline="") as fh:
    rows = list(csv.DictReader(fh))


def get(sex, cuts, key):
    m = {int(r["cutoff_age"]): r for r in rows if r["sex"] == sex}
    out = []
    for c in cuts:
        v = m.get(c, {}).get(key, "")
        out.append(float(v) if v not in ("", None) else np.nan)
    return np.array(out, dtype=float)


WA_CURVE = {
    "Male":   {16: 0.0403, 17: 0.2698, 18: 0.4306, 19: 0.4700, 20: 0.5576, 21: 0.6624,
               22: 0.7563, 23: 0.8250, 24: 0.8787, 25: 0.9119, 26: 0.9408},
    "Female": {16: 0.1980, 17: 0.2614, 18: 0.3553, 19: 0.4136, 20: 0.5074, 21: 0.6188,
               22: 0.7246, 23: 0.7842, 24: 0.8346, 25: 0.8728, 26: 0.8982},
}


def wa_line(sex, cuts):
    return np.array([WA_CURVE[sex].get(c, np.nan) for c in cuts], dtype=float)


plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans",
                     "axes.linewidth": 0.8, "legend.frameon": False})
C_FULL = "#0072B2"; C_A = "#D55E00"; C_B = "#009E73"; C_WA = "#CC79A7"

# ------------------------------------------------------------------ FIGURE 2
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
for ax, (sex, lab) in zip(axes, [("Male", "(A) Male"), ("Female", "(B) Female")]):
    cuts = list(range(16, 27))
    cx = np.array(cuts, dtype=float)
    full = get(sex, cuts, "r2_all")
    ga = get(sex, cuts, "r2_A_true_pred")
    gb = get(sex, cuts, "r2_B_trivial")
    pct = get(sex, cuts, "pct_A")
    wa = wa_line(sex, cuts)

    ax.fill_between(cx, ga, full, color="0.86", zorder=0)
    ax.plot(cx, full, color=C_FULL, ls="-", marker="o", ms=4, lw=1.6,
            label="Full sample")
    ax.plot(cx, ga, color=C_A, ls="--", marker="s", ms=4, lw=1.6, mfc="white",
            label="Group A (continuing)")
    ax.plot(cx, gb, color=C_B, ls=":", lw=1.4, label="Group B (career-ended)")
    mwa = np.isfinite(wa)
    ax.plot(cx[mwa], wa[mwa], color=C_WA, ls="--", marker="^", ms=4, lw=1.4,
            label="World Athletics ref.")

    # Group A percentages along the bottom
    for c, p in zip(cuts, pct):
        if not np.isfinite(p):
            continue
        ax.text(c, -0.055, f"{p:.0f}" if p >= 10 else f"{p:.1f}",
                ha="center", va="center", fontsize=5.4, color="0.42")
    ax.text(21, -0.108, "Group A, % of full sample", ha="center", va="center",
            fontsize=5.6, style="italic", color="0.42")

    ax.set_title(lab, fontsize=9, loc="left")
    ax.set_xlabel("Cutoff age (years)")
    ax.set_xlim(15.5, 26.5)
    ax.set_ylim(-0.14, 1.03)
    ax.set_xticks(range(16, 27, 2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", color="0.9", lw=0.6)

axes[0].set_ylabel("Cross-validated R²")
axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 0.135), fontsize=7.0,
               borderpad=0.3, labelspacing=0.35, handlelength=2.4)
fig.tight_layout()
fig.savefig(OUT / "Figure2_redraw.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "Figure2_redraw.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ FIGURE 3
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
for ax, (sex, lab) in zip(axes, [("Male", "(A) Male"), ("Female", "(B) Female")]):
    cuts = list(range(16, 23))
    cx = np.array(cuts, dtype=float)
    ga = get(sex, cuts, "r2_A_true_pred")
    lo = get(sex, cuts, "r2_A_ci_lo")
    hi = get(sex, cuts, "r2_A_ci_hi")
    nA = get(sex, cuts, "n_A")
    wa = wa_line(sex, cuts)

    ax.fill_between(cx, lo, hi, color=C_A, alpha=0.15, zorder=0)
    prim = cx <= 18
    supp = cx >= 18
    ax.plot(cx[supp], ga[supp], color=C_A, ls="--", lw=1.2, marker="s", ms=4,
            mfc=C_A, label="Group A R² (suppl. 19–22)")
    ax.plot(cx[prim], ga[prim], color=C_A, ls="-", lw=2.2, marker="s", ms=5,
            mfc="white", label="Group A R² (primary 16–18)")
    mwa = np.isfinite(wa)
    ax.plot(cx[mwa], wa[mwa], color=C_WA, ls="--", marker="^", ms=4, lw=1.4,
            label="World Athletics ref.")
    for c, y, n in zip(cuts, ga, nA):
        ax.annotate(f"n={int(n)}", (c, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=6, color="0.35")

    ax.set_title(lab, fontsize=9, loc="left")
    ax.set_xlabel("Cutoff age (years)")
    ax.set_xlim(15.5, 22.5)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(16, 23))
    ax.grid(axis="y", color="0.9", lw=0.6)

axes[0].set_ylabel("Cross-validated R² (Group A)")
h, l = axes[0].get_legend_handles_labels()
o = [l.index("Group A R² (primary 16–18)"), l.index("Group A R² (suppl. 19–22)"),
     l.index("World Athletics ref.")]
axes[0].legend([h[i] for i in o], [l[i] for i in o], loc="lower right",
               fontsize=6.8, borderpad=0.3, labelspacing=0.35)
fig.tight_layout()
fig.savefig(OUT / "Figure3_redraw.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "Figure3_redraw.pdf", bbox_inches="tight")
plt.close(fig)

print(f"OK: Figure2_redraw + Figure3_redraw (png+pdf) -> {OUT}")
