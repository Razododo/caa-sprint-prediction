"""
Step 9: Compare CAA results with WA paper results.

Input:  results/raw_results/model_results.csv (CAA results)
        data/external/wa_results_summary.csv (WA paper results)
Output: results/tables/comparison_summary.csv
        results/figures/ (comparison plots)
"""
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.io import load_config, ensure_dirs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    config = load_config()
    ensure_dirs(config)

    results_dir = Path(config["paths"]["raw_results"])
    tables_dir = Path(config["paths"]["tables"])
    figures_dir = Path(config["paths"]["figures"])
    external_dir = Path(config["paths"]["external"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load CAA results
    caa_path = results_dir / "model_results.csv"
    if not caa_path.exists():
        logger.error("model_results.csv not found. Run s06 first.")
        return

    df_caa = pd.read_csv(caa_path)
    caa_random = df_caa[df_caa["cv_strategy"] == "random"].copy()

    # Best model per cutoff (GradientBoosting, traj_wind)
    caa_best = caa_random[
        (caa_random["model"] == "GradientBoosting") &
        (caa_random["variant"] == "traj_wind")
    ].copy()

    # Load WA results if available
    wa_path = external_dir / "wa_results_summary.csv"
    has_wa = wa_path.exists()
    df_wa = pd.read_csv(wa_path) if has_wa else None

    # ===========================
    # Figure: Predictability curve
    # ===========================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    colors = config["plotting"]

    for i, (sex, ax) in enumerate(zip(["Male", "Female"], axes)):
        caa_sex = caa_best[caa_best["sex"] == sex].sort_values("cutoff_age")

        ax.plot(caa_sex["cutoff_age"], caa_sex["R2_mean"],
                "o-", color=colors[f"color_{'male' if sex == 'Male' else 'female'}"],
                label="CAA", markersize=5, linewidth=2)

        if "R2_ci_lo" in caa_sex.columns:
            ax.fill_between(
                caa_sex["cutoff_age"],
                caa_sex["R2_ci_lo"],
                caa_sex["R2_ci_hi"],
                alpha=0.2,
                color=colors[f"color_{'male' if sex == 'Male' else 'female'}"],
            )

        if has_wa and df_wa is not None:
            wa_rand = df_wa[df_wa["cv_strategy"] == "Random5Fold"]
            wa_sex = wa_rand[wa_rand["sex"] == sex].sort_values("cutoff_age")
            if len(wa_sex) > 0:
                ax.plot(wa_sex["cutoff_age"], wa_sex["R2"],
                        "s--", color=colors[f"color_wa_{'male' if sex == 'Male' else 'female'}"],
                        label="WA", markersize=4, linewidth=1.5)

        ax.set_xlabel("Cutoff Age")
        if i == 0:
            ax.set_ylabel("R²")
        ax.set_title(sex)
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Predictability Curve: R² by Cutoff Age", fontsize=12)
    fig.tight_layout()

    for fmt in config["plotting"]["save_formats"]:
        fig.savefig(figures_dir / f"fig2_predictability_curve.{fmt}", dpi=config["plotting"]["dpi"])
    plt.close(fig)
    logger.info("Saved fig2_predictability_curve")

    # ===========================
    # Figure: Model comparison (3 models × trajectory_only vs traj_wind)
    # ===========================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    _variant_label = {"trajectory_only": "trajectory-only", "traj_wind": "trajectory + wind"}
    for i, (sex, ax) in enumerate(zip(["Male", "Female"], axes)):
        for variant, ls in [("trajectory_only", "--"), ("traj_wind", "-")]:
            for model_name, color in [("Ridge", "#2ca02c"), ("RandomForest", "#ff7f0e"), ("GradientBoosting", "#1f77b4")]:
                sub = caa_random[
                    (caa_random["sex"] == sex) &
                    (caa_random["variant"] == variant) &
                    (caa_random["model"] == model_name)
                ].sort_values("cutoff_age")
                if len(sub) > 0:
                    lbl = f"{model_name} ({_variant_label[variant]})"
                    ax.plot(sub["cutoff_age"], sub["R2_mean"],
                            linestyle=ls, color=color, label=lbl, markersize=3, linewidth=1)

        ax.set_xlabel("Cutoff Age")
        if i == 0:
            ax.set_ylabel("R²")
        ax.set_title(sex)
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(0.8, 1.01)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Model × Variant Comparison", fontsize=12)
    fig.tight_layout()
    for fmt in config["plotting"]["save_formats"]:
        fig.savefig(figures_dir / f"fig_model_comparison.{fmt}", dpi=config["plotting"]["dpi"])
    plt.close(fig)
    logger.info("Saved fig_model_comparison")

    # ===========================
    # Comparison summary table
    # ===========================
    summary = caa_best[["cutoff_age", "sex", "R2_mean", "RMSE_mean", "n_samples"]].copy()
    summary.columns = ["cutoff_age", "sex", "CAA_R2", "CAA_RMSE", "CAA_n"]

    if has_wa and df_wa is not None:
        wa_rand = df_wa[df_wa["cv_strategy"] == "Random5Fold"]
        summary = summary.merge(
            wa_rand[["cutoff_age", "sex", "R2", "RMSE"]].rename(
                columns={"R2": "WA_R2", "RMSE": "WA_RMSE"}
            ),
            on=["cutoff_age", "sex"],
            how="left",
        )
        summary["delta_R2"] = summary["CAA_R2"] - summary["WA_R2"]

    summary.to_csv(tables_dir / "comparison_summary.csv", index=False)
    logger.info(f"Saved comparison_summary.csv")

    logger.info("\n" + "=" * 60)
    logger.info("CAA RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(summary.to_string(index=False))


if __name__ == "__main__":
    main()
