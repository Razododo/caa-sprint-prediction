"""
Step 15: Generate all publication figures.

Fig 2 — Population Composition Decomposition (HERO)
Fig 3 — Controlled Developmental Curve (Group A)
Fig 4 — Anthropometric Ablation
Fig 5 — Attrition Curve
Fig 6 — PB-Stratified R²
Fig 7 — Province-Disjoint Validation
Fig 8 — Permutation importance by feature group (cutoff 18, Group A)
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import load_config, ensure_dirs

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Display labels for variants and feature groups (charts only)
LABEL_VARIANT = {
    "trajectory_only": "trajectory-only",
    "traj_wind": "trajectory + wind",
    "full_no_anthro": "full (no anthropometry)",
}
LABEL_FEATURE_GROUP = {
    "trajectory_raw": "raw trajectory",
    "trajectory_wc": "wind-corrected trajectory",
    "career_structure": "career structure",
    "round_performance": "round performance",
    "trajectory_dynamics": "trajectory dynamics",
    "best_time_raw": "best time",
    "improvement_slope_raw": "improvement slope",
    "wind": "wind",
    "era": "era",
}


def _save(fig: plt.Figure, fig_dir: Path, name: str, cfg: dict) -> None:
    for fmt in cfg["save_formats"]:
        fig.savefig(fig_dir / f"{name}.{fmt}", dpi=cfg["dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {name}")


# ======================================================================
# Fig 2 — Population Composition Decomposition (HERO)
# ======================================================================
def fig2_composition(tables: Path, fig_dir: Path, cfg: dict) -> None:
    logger.info("Fig 2: Population Composition Decomposition")
    df = pd.read_csv(tables / "composition_decomposition.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for i, (sex, ax) in enumerate(zip(["Male", "Female"], axes)):
        sub = df[df["sex"] == sex].sort_values("cutoff_age")
        ages = sub["cutoff_age"].values
        r2_full = sub["r2_all"].values
        r2_a = sub["r2_A_true_pred"].values
        r2_b = sub["r2_B_trivial"].values
        wa = sub["wa_r2"].values
        ci_lo = sub["r2_A_ci_lo"].values
        ci_hi = sub["r2_A_ci_hi"].values
        pct_a = sub["pct_A"].values

        # Shaded inflation region
        ax.fill_between(ages, r2_a, r2_full, alpha=0.15,
                         color=cfg[f"color_{'male' if sex == 'Male' else 'female'}"],
                         label="Composition inflation")

        # R² lines
        ax.plot(ages, r2_full, "o-",
                color=cfg[f"color_{'male' if sex == 'Male' else 'female'}"],
                markersize=5, linewidth=2, label=r"$R^2_{full}$", zorder=5)
        ax.plot(ages, r2_a, "s-",
                color=cfg[f"color_{'male' if sex == 'Male' else 'female'}"],
                markersize=5, linewidth=1.5, markerfacecolor="white",
                markeredgewidth=1.5, label=r"$R^2_{Group\ A}$", zorder=5)
        ax.fill_between(ages, ci_lo, ci_hi, alpha=0.12,
                         color=cfg[f"color_{'male' if sex == 'Male' else 'female'}"])
        ax.plot(ages, r2_b, ":", color="grey", linewidth=1,
                label=r"$R^2_{Group\ B}$", zorder=3)

        # WA reference
        wa_valid = ~np.isnan(wa)
        if wa_valid.any():
            ax.plot(ages[wa_valid], wa[wa_valid], "^--",
                    color=cfg[f"color_wa_{'male' if sex == 'Male' else 'female'}"],
                    markersize=4, linewidth=1.5, label="WA reference", zorder=4)

        # Group A % annotations at bottom
        for age, pct in zip(ages, pct_a):
            if pct >= 1:
                ax.annotate(f"{pct:.0f}%", (age, 0.03), fontsize=6,
                            ha="center", va="bottom", color="grey")

        ax.set_xlabel("Cutoff Age", fontsize=cfg["font_size"])
        if i == 0:
            ax.set_ylabel(r"$R^2$", fontsize=cfg["font_size"])
        ax.set_ylim(0, 1.08)
        ax.set_xlim(15.5, 26.5)
        ax.text(0.02, 0.95, f"({'A' if i == 0 else 'B'}) {sex}",
                transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ax.legend(fontsize=7, loc="center right")
        ax.grid(True, alpha=0.2)
        ax.text(0.5, 0.08, "Group A %", transform=ax.transAxes,
                fontsize=6, ha="center", color="grey")

    fig.tight_layout()
    _save(fig, fig_dir, "fig2_composition_decomposition", cfg)


# ======================================================================
# Fig 3 — Controlled Developmental Curve
# ======================================================================
def fig3_developmental(tables: Path, results_dir: Path, fig_dir: Path, cfg: dict) -> None:
    logger.info("Fig 3: Controlled Developmental Curve")
    s11 = pd.read_csv(results_dir / "controlled_analysis" / "group_a_results.csv")
    gb_tw = s11[
        (s11["model"] == "GradientBoosting")
        & (s11["variant"] == "traj_wind")
        & (s11["cv_strategy"] == "random")
    ]

    wa = pd.read_csv(Path("data/external/wa_results_summary.csv"))
    wa_rand = wa[wa["cv_strategy"] == "Random5Fold"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for i, (sex, ax) in enumerate(zip(["Male", "Female"], axes)):
        sub = gb_tw[gb_tw["sex"] == sex].sort_values("cutoff_age")

        primary = sub[sub["cutoff_age"] <= 18]
        supp = sub[sub["cutoff_age"] > 18]

        c = cfg[f"color_{'male' if sex == 'Male' else 'female'}"]

        # Primary (solid)
        ax.plot(primary["cutoff_age"], primary["R2_mean"], "o-",
                color=c, markersize=6, linewidth=2, zorder=5)
        if "R2_ci_lo" in primary.columns:
            ax.fill_between(primary["cutoff_age"],
                            primary["R2_ci_lo"], primary["R2_ci_hi"],
                            alpha=0.2, color=c)

        # Supplementary (dashed)
        if len(supp) > 0:
            ax.plot(supp["cutoff_age"], supp["R2_mean"], "o--",
                    color=c, markersize=4, linewidth=1.5, alpha=0.6, zorder=4)
            if "R2_ci_lo" in supp.columns:
                ax.fill_between(supp["cutoff_age"],
                                supp["R2_ci_lo"], supp["R2_ci_hi"],
                                alpha=0.1, color=c)

        # Connect primary to supplementary
        if len(supp) > 0:
            bridge = pd.concat([primary.tail(1), supp.head(1)])
            ax.plot(bridge["cutoff_age"], bridge["R2_mean"], "--",
                    color=c, linewidth=1, alpha=0.4)

        # Annotate n
        for _, r in sub.iterrows():
            ax.annotate(f"n={int(r['n_samples'])}",
                        (r["cutoff_age"], r["R2_mean"] - 0.03),
                        fontsize=6, ha="center", color="grey")

        # WA reference
        wa_sex = wa_rand[wa_rand["sex"] == sex].sort_values("cutoff_age")
        wa_in_range = wa_sex[(wa_sex["cutoff_age"] >= 16) & (wa_sex["cutoff_age"] <= 22)]
        if len(wa_in_range) > 0:
            ax.plot(wa_in_range["cutoff_age"], wa_in_range["R2"], "^--",
                    color=cfg[f"color_wa_{'male' if sex == 'Male' else 'female'}"],
                    markersize=4, linewidth=1.5, label="WA reference", zorder=3)

        ax.set_xlabel("Cutoff Age", fontsize=cfg["font_size"])
        if i == 0:
            ax.set_ylabel(r"$R^2$ (Group A)", fontsize=cfg["font_size"])
        ax.set_ylim(0.25, 0.95)
        ax.set_xlim(15.5, 22.5)
        ax.text(0.02, 0.95, f"({'A' if i == 0 else 'B'}) {sex}",
                transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    _save(fig, fig_dir, "fig3_developmental_curve", cfg)


# ======================================================================
# Fig 4 — Anthropometric Ablation
# ======================================================================
def fig4_anthro(tables: Path, fig_dir: Path, cfg: dict) -> None:
    logger.info("Fig 4: Anthropometric Ablation")
    df = pd.read_csv(tables / "table3_anthro_ablation.csv")

    fig, ax = plt.subplots(figsize=(6, 4))

    cutoffs = sorted(df["cutoff_age"].unique())
    width = 0.35
    x = np.arange(len(cutoffs))

    for j, (sex, color) in enumerate([
        ("Male", cfg["color_male"]),
        ("Female", cfg["color_female"]),
    ]):
        sub = df[df["sex"] == sex].sort_values("cutoff_age")
        delta = sub["anthro_delta_R2"].values

        # CI from V2 and V3 CI bounds
        ci_lo_v2 = sub["R2_v2_ci_lo"].values
        ci_hi_v2 = sub["R2_v2_ci_hi"].values
        ci_lo_v3 = sub["R2_v3_ci_lo"].values
        ci_hi_v3 = sub["R2_v3_ci_hi"].values
        err_lo = delta - (ci_lo_v3 - ci_hi_v2)
        err_hi = (ci_hi_v3 - ci_lo_v2) - delta
        err = np.array([np.abs(err_lo), np.abs(err_hi)])

        offset = -width / 2 + j * width
        bars = ax.bar(x + offset, delta, width * 0.9, color=color,
                      alpha=0.7, label=sex, edgecolor=color)
        ax.errorbar(x + offset, delta, yerr=err,
                    fmt="none", color="black", capsize=3, linewidth=0.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in cutoffs])
    ax.set_xlabel("Cutoff Age", fontsize=cfg["font_size"])
    ax.set_ylabel(r"$\Delta R^2$ (with anthro − without)", fontsize=cfg["font_size"])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(-0.08, 0.04)

    fig.tight_layout()
    _save(fig, fig_dir, "fig4_anthro_ablation", cfg)


# ======================================================================
# Fig 5 — Attrition Curve
# ======================================================================
def fig5_attrition(tables: Path, fig_dir: Path, cfg: dict) -> None:
    logger.info("Fig 5: Attrition Curve")
    df = pd.read_csv(tables / "attrition_curve.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for i, (sex, ax) in enumerate(zip(["Male", "Female"], axes)):
        sub = df[df["sex"] == sex].sort_values("age")
        ages = sub["age"].values
        active = sub["active_athletes"].values
        dropout = sub["dropout_rate"].values * 100

        # Area fill for active athletes
        ax.fill_between(ages, 0, active, alpha=0.3,
                         color=cfg[f"color_{'male' if sex == 'Male' else 'female'}"])
        ax.plot(ages, active, "-",
                color=cfg[f"color_{'male' if sex == 'Male' else 'female'}"],
                linewidth=2, label="Active athletes")
        ax.set_xlabel("Age", fontsize=cfg["font_size"])
        if i == 0:
            ax.set_ylabel("Active Athletes", fontsize=cfg["font_size"])

        # Secondary axis: dropout rate
        ax2 = ax.twinx()
        ax2.plot(ages, dropout, "o-", color="#ff7f0e", markersize=3,
                 linewidth=1.5, label="Dropout rate", zorder=5)
        if i == 1:
            ax2.set_ylabel("Dropout Rate (%)", fontsize=cfg["font_size"], color="#ff7f0e")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")
        ax2.set_ylim(0, 105)

        # Peak dropout: max in 14–32, but require active_athletes >= 5 so tiny-n (e.g. age 31 n=1) is not labeled
        valid = sub[(sub["age"] >= 14) & (sub["age"] <= 32) & (sub["active_athletes"] >= 5)]
        if len(valid) > 0:
            peak_idx = valid["dropout_rate"].idxmax()
            peak_row = sub.loc[peak_idx]
            # Female (B): put text left of point, arrow points right to peak (short, like Male)
            if i == 1:
                ax2.annotate(
                    f"Peak: age {int(peak_row['age'])} ({peak_row['dropout_rate']*100:.0f}%)",
                    (peak_row["age"], peak_row["dropout_rate"] * 100),
                    textcoords="offset points", xytext=(-22, -5),
                    fontsize=7, color="#ff7f0e", ha="right",
                    arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=0.8),
                )
            else:
                ax2.annotate(
                    f"Peak: age {int(peak_row['age'])}\n({peak_row['dropout_rate']*100:.0f}%)",
                    (peak_row["age"], peak_row["dropout_rate"] * 100),
                    textcoords="offset points", xytext=(15, -10),
                    fontsize=7, color="#ff7f0e",
                    arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=0.8),
                )

        ax.set_xlim(12, 32)
        ax.text(0.02, 0.95, f"({'A' if i == 0 else 'B'}) {sex}",
                transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    _save(fig, fig_dir, "fig5_attrition_curve", cfg)


# ======================================================================
# Fig 6 — PB-Stratified R² Analysis
# ======================================================================
def fig6_pb_stratified(results_dir: Path, fig_dir: Path, cfg: dict) -> None:
    logger.info("Fig 6: PB-Stratified R²")
    df = pd.read_csv(results_dir / "controlled_analysis" / "pb_stratified_results.csv")
    df = df.dropna(subset=["R2_mean"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for i, (sex, ax) in enumerate(zip(["Male", "Female"], axes)):
        sub = df[df["sex"] == sex]
        cutoffs = sorted(sub["cutoff_age"].unique())

        strata = sub["stratum"].unique()
        colors = {"elite": "#2ca02c", "sub-elite": "#ff7f0e", "grassroots": "#9467bd"}
        width = 0.25
        x = np.arange(len(cutoffs))

        for j, stratum in enumerate(["elite", "sub-elite", "grassroots"]):
            s = sub[sub["stratum"] == stratum].sort_values("cutoff_age")
            if len(s) == 0:
                continue
            vals = []
            errs_lo = []
            errs_hi = []
            positions = []
            for k, cut in enumerate(cutoffs):
                row = s[s["cutoff_age"] == cut]
                if len(row) > 0:
                    r = row.iloc[0]
                    vals.append(r["R2_mean"])
                    lo = r["R2_mean"] - r.get("R2_ci_lo", r["R2_mean"])
                    hi = r.get("R2_ci_hi", r["R2_mean"]) - r["R2_mean"]
                    errs_lo.append(abs(lo))
                    errs_hi.append(abs(hi))
                    positions.append(k)

            offset = -width + j * width
            pos_arr = np.array(positions)
            ax.bar(pos_arr + offset, vals, width * 0.9,
                   color=colors.get(stratum, "grey"), alpha=0.7,
                   label=stratum, edgecolor=colors.get(stratum, "grey"))
            ax.errorbar(pos_arr + offset, vals,
                        yerr=[errs_lo, errs_hi],
                        fmt="none", color="black", capsize=2, linewidth=0.6)

            # Annotate n and SD
            for p, v, idx in zip(positions, vals, range(len(vals))):
                row = s[s["cutoff_age"] == cutoffs[p]]
                if len(row) > 0:
                    n = int(row.iloc[0]["n"])
                    sd = row.iloc[0].get("pb_sd", 0)
                    ax.text(p + offset, max(v + 0.03, 0.05),
                            f"n={n}\nSD={sd:.2f}", fontsize=5,
                            ha="center", va="bottom")

        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in cutoffs])
        ax.set_xlabel("Cutoff Age", fontsize=cfg["font_size"])
        if i == 0:
            ax.set_ylabel(r"$R^2$", fontsize=cfg["font_size"])
        ax.set_ylim(-0.3, 1.0)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
        ax.text(0.02, 0.95, f"({'A' if i == 0 else 'B'}) {sex}",
                transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    _save(fig, fig_dir, "fig6_pb_stratified", cfg)


# ======================================================================
# Fig 7 — Province-Disjoint Validation
# ======================================================================
def fig7_province_disjoint(results_dir: Path, fig_dir: Path, cfg: dict) -> None:
    logger.info("Fig 7: Province-Disjoint Validation")
    s11 = pd.read_csv(results_dir / "controlled_analysis" / "group_a_results.csv")

    gb_fw = s11[
        (s11["model"] == "GradientBoosting")
        & (s11["variant"] == "full_no_anthro")
    ]
    random_cv = gb_fw[gb_fw["cv_strategy"] == "random"]
    prov_cv = gb_fw[gb_fw["cv_strategy"] == "province_disjoint"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    cutoffs = [16, 17, 18]
    x_pos = 0

    for sex, color in [("Male", cfg["color_male"]), ("Female", cfg["color_female"])]:
        for cut in cutoffs:
            r_rand = random_cv[(random_cv["sex"] == sex) & (random_cv["cutoff_age"] == cut)]
            r_prov = prov_cv[(prov_cv["sex"] == sex) & (prov_cv["cutoff_age"] == cut)]

            if len(r_rand) == 0 or len(r_prov) == 0:
                x_pos += 1
                continue

            y_rand = r_rand.iloc[0]["R2_mean"]
            y_prov = r_prov.iloc[0]["R2_mean"]
            delta = y_prov - y_rand

            ax.plot([x_pos, x_pos], [y_rand, y_prov], "-", color=color,
                    linewidth=2, alpha=0.6)
            ax.plot(x_pos, y_rand, "o", color=color, markersize=8, zorder=5)
            ax.plot(x_pos, y_prov, "s", color=color, markersize=8,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=2, zorder=5)

            mid = (y_rand + y_prov) / 2
            ax.annotate(f"Δ={delta:+.03f}", (x_pos + 0.15, mid),
                        fontsize=7, color=color)

            x_pos += 1
        x_pos += 0.5

    # X labels
    labels = []
    for sex in ["Male", "Female"]:
        for cut in cutoffs:
            labels.append(f"{sex[0]}{cut}")

    tick_positions = list(range(len(cutoffs))) + [x + 3.5 for x in range(len(cutoffs))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"M{c}" for c in cutoffs] + [f"F{c}" for c in cutoffs],
                       fontsize=8)

    ax.set_ylabel(r"$R^2$", fontsize=cfg["font_size"])
    ax.set_xlabel("Sex × Cutoff Age", fontsize=cfg["font_size"])
    ax.grid(True, alpha=0.2, axis="y")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="grey", label="Random CV",
               markersize=7, linestyle="None"),
        Line2D([0], [0], marker="s", color="grey", label="Province-disjoint",
               markerfacecolor="white", markeredgecolor="grey",
               markersize=7, linestyle="None", markeredgewidth=2),
        Line2D([0], [0], color=cfg["color_male"], label="Male", linewidth=2),
        Line2D([0], [0], color=cfg["color_female"], label="Female", linewidth=2),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    fig.tight_layout()
    _save(fig, fig_dir, "fig7_province_disjoint", cfg)


# ======================================================================
# Fig 8 — Permutation importance by feature group
# ======================================================================
def fig8_permutation_importance(results_dir: Path, fig_dir: Path, cfg: dict) -> None:
    logger.info("Fig 8: Permutation importance by feature group")
    path = results_dir / "controlled_analysis" / "permutation_importance_by_group.csv"
    if not path.exists():
        logger.warning(f"  Skip: {path} not found")
        return
    df = pd.read_csv(path)
    # Order groups by mean importance (desc) for consistent display
    order = (
        df.groupby("feature_group")["r2_drop_mean"]
        .apply(lambda x: x.max())
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    colors = [cfg["color_male"], cfg["color_female"]]
    for i, (sex, ax) in enumerate(zip(["Male", "Female"], axes)):
        sub = df[df["sex"] == sex].copy()
        sub["feature_group"] = pd.Categorical(sub["feature_group"], categories=order, ordered=True)
        sub = sub.sort_values("feature_group", ascending=True)
        y_pos = np.arange(len(sub))
        vals = sub["r2_drop_mean"].values
        errs = sub["r2_drop_std"].values
        bars = ax.barh(y_pos, vals, height=0.65, color=colors[i], alpha=0.85, edgecolor="none")
        # Error bars: same hue as bar, slightly darker for visibility
        ax.errorbar(
            vals, y_pos, xerr=errs, fmt="none",
            ecolor=colors[i], elinewidth=1.2, capsize=2.5, capthick=1.0,
        )
        ax.axvline(0, color="black", linewidth=0.6, linestyle="-")
        ax.set_yticks(y_pos)
        y_labels = [LABEL_FEATURE_GROUP.get(g, g) for g in sub["feature_group"].values]
        ax.set_yticklabels(y_labels, fontsize=cfg["font_size"] - 1)
        ax.set_xlabel(r"$R^2$ drop (permutation importance)", fontsize=cfg["font_size"])
        if i == 0:
            ax.set_ylabel("Feature group", fontsize=cfg["font_size"])
        ax.text(0.02, 0.95, f"({'A' if i == 0 else 'B'}) {sex}",
                transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    _save(fig, fig_dir, "fig8_permutation_importance", cfg)


# ======================================================================
# Main
# ======================================================================
def main() -> None:
    config = load_config()
    ensure_dirs(config)

    cfg = config["plotting"]
    tables = Path(config["paths"]["tables"])
    results_dir = Path(config["paths"]["raw_results"])
    fig_dir = Path(config["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": cfg["font_size"],
        "axes.labelsize": cfg["font_size"],
        "xtick.labelsize": cfg["font_size"] - 1,
        "ytick.labelsize": cfg["font_size"] - 1,
    })

    fig2_composition(tables, fig_dir, cfg)
    fig3_developmental(tables, results_dir, fig_dir, cfg)
    fig4_anthro(tables, fig_dir, cfg)
    fig5_attrition(tables, fig_dir, cfg)
    fig6_pb_stratified(results_dir, fig_dir, cfg)
    fig7_province_disjoint(results_dir, fig_dir, cfg)
    fig8_permutation_importance(results_dir, fig_dir, cfg)

    logger.info(f"\nAll figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
