"""Redraw Figures 4 and 5.

  Fig 4  Permutation importance by feature group at cutoff age 18 (Group A).
  Fig 5  Random 5-fold vs province-disjoint cross-validation, Group A,
         cutoff ages 16-18, both sexes.

Input:   results/raw_results/controlled_analysis/permutation_importance_by_group.csv
         results/raw_results/controlled_analysis/group_a_results.csv
         Both are produced by src/s11_controlled_analysis.py and
         scripts/run_permutation_and_parsimonious_group_a.py. They live under
         results/raw_results/, which is excluded from this repository (see
         .gitignore), so this script is shipped as provenance for the two
         figures rather than as a stand-alone reproduction step. Run the
         pipeline first, or request the raw-result files from the authors.
Output:  results/figures/Figure{4,5}_redraw.{png,pdf}
Run:     python redraw_figures_4_5.py
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "raw_results" / "controlled_analysis"
OUT = HERE / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PERM_CSV = RAW / "permutation_importance_by_group.csv"
GA_CSV = RAW / "group_a_results.csv"
for p in (PERM_CSV, GA_CSV):
    if not p.exists():
        raise SystemExit(
            f"missing input: {p}\n"
            "results/raw_results/ is not distributed with this repository "
            "(see .gitignore). Run the pipeline to regenerate it, or contact "
            "the corresponding author."
        )


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans",
                     "axes.linewidth": 0.8, "legend.frameon": False})
CM = "#0072B2"; CF = "#D62728"

# ---- FIGURE 4: permutation importance at cutoff 18 ----
perm = read(PERM_CSV)
LAB = {"trajectory_raw": "Raw trajectory", "trajectory_wc": "Wind-corrected trajectory",
       "trajectory_dynamics": "Trajectory dynamics", "career_structure": "Career structure",
       "round_performance": "Round performance", "wind": "Wind", "era": "Era"}


def pdata(sex):
    return {r["feature_group"]: (float(r["r2_drop_mean"]), float(r["r2_drop_std"]))
            for r in perm if r["sex"] == sex}


order = ["trajectory_raw", "trajectory_wc", "career_structure", "era", "wind",
         "trajectory_dynamics", "round_performance"]
order = order[::-1]  # bottom-to-top ascending
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharex=True)
for ax, (sex, lab, c, n) in zip(axes, [("Male", "(A) Male (n = 2,662)", CM, 2662),
                                       ("Female", "(B) Female (n = 522)", CF, 522)]):
    d = pdata(sex); y = np.arange(len(order))
    means = [d[g][0] for g in order]; stds = [d[g][1] for g in order]
    ax.barh(y, means, xerr=stds, color=c, alpha=0.85, ecolor="0.4", capsize=2,
            height=0.62, error_kw={"lw": 0.8})
    ax.axvline(0, color="0.5", lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels([LAB[g] for g in order], fontsize=7.5)
    ax.set_title(lab, fontsize=9, loc="left")
    ax.set_xlabel("Mean R² drop when permuted")
    ax.grid(axis="x", color="0.9", lw=0.6)
axes[0].tick_params(labelleft=True); axes[1].tick_params(labelleft=False)
fig.text(0.5, -0.02, "Error bars = ± 1 SD across cross-validation folds",
         ha="center", fontsize=7, style="italic")
fig.tight_layout()
fig.savefig(OUT / "Figure4_redraw.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "Figure4_redraw.pdf", bbox_inches="tight")
plt.close(fig)

# ---- FIGURE 5: random vs province-disjoint ----
ga = read(GA_CSV)


def val(cut, sex, cv):
    for r in ga:
        if (r["variant"] == "traj_wind" and r["model"] == "GradientBoosting"
                and r["cutoff_age"] == str(cut) and r["sex"] == sex
                and r["cv_strategy"] == cv):
            return float(r["R2_mean"])
    return None


conds = [(16, "Male"), (17, "Male"), (18, "Male"),
         (16, "Female"), (17, "Female"), (18, "Female")]
fig, ax = plt.subplots(figsize=(6.6, 3.6))
xs = np.arange(len(conds))
for i, (cut, sex) in enumerate(conds):
    rnd = val(cut, sex, "random"); pdj = val(cut, sex, "province_disjoint")
    d = pdj - rnd
    c = CM if sex == "Male" else CF
    ax.plot([i, i], [rnd, pdj], color=c, lw=1.2, zorder=1)
    ax.plot(i, rnd, marker="o", ms=7, color=c, zorder=2)
    ax.plot(i, pdj, marker="s", ms=7, mfc="white", mec=c, mew=1.5, zorder=2)
    ax.annotate(f"ΔR²={d:+.3f}", (i, min(rnd, pdj) - 0.005), ha="center",
                va="top", fontsize=6.5, color="0.3")
ax.set_xticks(xs)
ax.set_xticklabels([f"{'M' if s == 'Male' else 'F'}{c}" for c, s in conds])
ax.set_ylabel("Cross-validated R² (Group A)"); ax.set_xlabel("Sex × cutoff age")
ax.set_xlim(-0.42, 5.42)
ax.set_ylim(0.45, 0.9); ax.grid(axis="y", color="0.9", lw=0.6)
leg = [Line2D([0], [0], marker="o", color="0.3", lw=0, ms=7, label="Random 5-fold CV"),
       Line2D([0], [0], marker="s", color="0.3", lw=0, ms=7, mfc="white", mew=1.5,
              label="Province-disjoint CV"),
       Line2D([0], [0], marker="o", color=CM, lw=0, ms=7, label="Male"),
       Line2D([0], [0], marker="o", color=CF, lw=0, ms=7, label="Female")]
ax.legend(handles=leg, loc="lower right", fontsize=7.2, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "Figure5_redraw.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "Figure5_redraw.pdf", bbox_inches="tight")
plt.close(fig)
print(f"OK: Figure4_redraw + Figure5_redraw (png+pdf) -> {OUT}")
