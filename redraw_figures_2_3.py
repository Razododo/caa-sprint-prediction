"""Redraw Figures 2 and 3 to match the revised manuscript captions.

Revision note. Earlier versions of this script drew a World Athletics reference
series as a dashed line with triangle markers, and carried that series as a
hard-coded WA_CURVE dictionary. Following Reviewer 4's third major comment the
quantitative World Athletics comparison has been removed from the manuscript,
so the series, the legend entry and the hard-coded values are removed here as
well. Leaving them in the script would have kept the removed numbers in the
public repository.

Figure 2  Full sample, Group A and Group B R² across cutoff ages 16 to 26, with
          the shaded band showing the full-sample minus Group A difference and
          the Group A percentage of the full sample along the bottom axis.

Figure 3  Group A R² across cutoff ages 16 to 22 under the published outcome
          (lifetime personal best), together with the prospective outcome
          (best performance strictly after the cutoff) at the three primary
          cutoffs. Showing both series makes the effect of the outcome
          definition visible in the figure rather than only in the text.

Input:   results/tables/composition_decomposition.csv
         results/tables/r4_post_cutoff_outcome.csv    (Figure 3 prospective series)
         results/tables/group_a_cutoff_sweep.csv      (Figure 3 lifetime-PB series;
             falls back to results/raw_results/controlled_analysis/group_a_results.csv,
             which is the same file under its working-tree name)
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
POST = HERE / "results" / "tables" / "r4_post_cutoff_outcome.csv"

# Figure 3 needs the Group A cutoff sweep at ages 16 to 22. In the working tree
# that series lives with the rest of the s11 output under results/raw_results/,
# which the public repository does not ship. A copy of the same file is kept in
# results/tables/, which is shipped, so the figure can be redrawn from the
# published package alone. Prefer the published copy and fall back to the raw
# one, so that the script is byte-identical in both repositories.
GA_PUBLISHED = HERE / "results" / "tables" / "group_a_cutoff_sweep.csv"
GA_RAW = HERE / "results" / "raw_results" / "controlled_analysis" / "group_a_results.csv"
GA = GA_PUBLISHED if GA_PUBLISHED.exists() else GA_RAW

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


def group_a_main(sex, cuts, key="R2_mean"):
    """Group A series from the same s11 run that produced Main Table 2.

    Figure 3 must not be drawn from composition_decomposition.csv: that file's
    r2_A_true_pred column comes from the diagnostic script, which uses a reduced
    feature set and a different seed base, so it differs from Main Table 2 by up
    to 0.036. The rows below reproduce Main Table 2 exactly at cutoffs 16 to 18
    and extend the same specification to 19 to 22 at 100 repeats.
    """
    if not GA.exists():
        raise SystemExit(f"missing input table: {GA}")
    with open(GA, newline="") as fh:
        gr = [r for r in csv.DictReader(fh)
              if r["variant"] == "traj_wind" and r["model"] == "GradientBoosting"
              and r["cv_strategy"] == "random"]
    m = {(r["sex"], int(float(r["cutoff_age"]))): r for r in gr}
    out = []
    for c in cuts:
        r = m.get((sex, c))
        out.append(float(r[key]) if r and r.get(key) not in ("", None) else np.nan)
    return np.array(out, dtype=float)


def post_cutoff(sex, cuts):
    """Prospective-outcome R², Gradient Boosting. NaN where not computed."""
    if not POST.exists():
        return np.full(len(cuts), np.nan)
    with open(POST, newline="") as fh:
        pr = [r for r in csv.DictReader(fh)
              if r["model"] == "GradientBoosting" and r["outcome"] == "pb_post_cutoff"]
    m = {(r["sex"], int(r["cutoff_age"])): float(r["R2_mean"]) for r in pr}
    return np.array([m.get((sex, c), np.nan) for c in cuts], dtype=float)


plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans",
                     "axes.linewidth": 0.8, "legend.frameon": False})
C_FULL = "#0072B2"; C_A = "#D55E00"; C_B = "#009E73"; C_POST = "#56B4E9"

# ------------------------------------------------------------------ FIGURE 2
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
for ax, (sex, lab) in zip(axes, [("Male", "(A) Male"), ("Female", "(B) Female")]):
    cuts = list(range(16, 27))
    cx = np.array(cuts, dtype=float)
    full = get(sex, cuts, "r2_all")
    ga = get(sex, cuts, "r2_A_true_pred")
    gb = get(sex, cuts, "r2_B_trivial")
    pct = get(sex, cuts, "pct_A")

    ax.fill_between(cx, ga, full, color="0.86", zorder=0)
    ax.plot(cx, full, color=C_FULL, ls="-", marker="o", ms=4, lw=1.6,
            label="Full sample")
    ax.plot(cx, ga, color=C_A, ls="--", marker="s", ms=4, lw=1.6, mfc="white",
            label="Group A (continuing)")
    ax.plot(cx, gb, color=C_B, ls=":", lw=1.4, label="Group B (career-ended)")

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
    ga = group_a_main(sex, cuts, "R2_mean")
    lo = group_a_main(sex, cuts, "R2_ci_lo")
    hi = group_a_main(sex, cuts, "R2_ci_hi")
    nA = group_a_main(sex, cuts, "n_samples")
    pc = post_cutoff(sex, cuts)

    ax.fill_between(cx, lo, hi, color=C_A, alpha=0.15, zorder=0)
    prim = cx <= 18
    supp = cx >= 18
    ax.plot(cx[supp], ga[supp], color=C_A, ls="--", lw=1.2, marker="s", ms=4,
            mfc=C_A, label="Lifetime PB, suppl. 19-22")
    ax.plot(cx[prim], ga[prim], color=C_A, ls="-", lw=2.2, marker="s", ms=5,
            mfc="white", label="Lifetime PB, primary 16-18")
    mp = np.isfinite(pc)
    if mp.any():
        ax.plot(cx[mp], pc[mp], color=C_POST, ls="-", lw=2.0, marker="D", ms=4.5,
                mfc="white", label="Post-cutoff best, 16-18")
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
order = ["Lifetime PB, primary 16-18", "Lifetime PB, suppl. 19-22",
         "Post-cutoff best, 16-18"]
o = [l.index(x) for x in order if x in l]
axes[0].legend([h[i] for i in o], [l[i] for i in o], loc="lower right",
               fontsize=6.8, borderpad=0.3, labelspacing=0.35)
fig.tight_layout()
fig.savefig(OUT / "Figure3_redraw.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "Figure3_redraw.pdf", bbox_inches="tight")
plt.close(fig)

print(f"OK: Figure2_redraw + Figure3_redraw (png+pdf) -> {OUT}")
