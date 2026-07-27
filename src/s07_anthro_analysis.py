"""
Step 7: Anthropometric subsample analysis.

Three-way comparison (see CURSOR_RULES.md):
1. Full sample, NO anthro features → main results (from Step 6)
2. Anthro subsample, NO anthro features → selection bias check
3. Anthro subsample, WITH anthro features → anthro contribution
ΔR² = (version 3) - (version 2)  ← this is the correct comparison
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from utils.io import load_config, ensure_dirs, read_parquet
from utils.validation import random_cv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_feature_registry() -> dict:
    project_root = Path(__file__).resolve().parent.parent
    return yaml.safe_load((project_root / "config" / "feature_registry.yaml").read_text())


def get_feature_columns(registry: dict, variant: str) -> list[str]:
    variant_def = registry["dataset_variants"][variant]
    cols = []
    for group_name in variant_def["groups"]:
        group = registry["groups"][group_name]
        cols.extend(f["name"] for f in group["features"])
    return cols


def get_models(config: dict) -> dict:
    mc = config["modeling"]["algorithms"]
    return {
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=mc["ridge"]["alpha"])),
        ]),
        "GradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(**mc["gradient_boosting"])),
        ]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    config = load_config()
    ensure_dirs(config)
    registry = load_feature_registry()

    cutoff_ages = config["features"]["cutoff_ages"]
    n_repeats = config["modeling"]["cv"]["n_repeats_quick"] if args.quick else config["modeling"]["cv"]["n_repeats"]
    n_folds = config["modeling"]["cv"]["n_folds"]
    seed = config["modeling"]["cv"]["random_seed"]
    min_anthro = config["anthropometric"]["min_subsample_for_analysis"]
    processed = Path(config["paths"]["processed"])
    results_dir = Path(config["paths"]["raw_results"])
    tables_dir = Path(config["paths"]["tables"])
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    traj_cols = get_feature_columns(registry, "trajectory_only")
    anthro_cols = get_feature_columns(registry, "traj_anthro")

    all_results = []

    for cutoff in cutoff_ages:
        feat_path = processed / f"features_cutoff_{cutoff}.parquet"
        if not feat_path.exists():
            continue

        df = read_parquet(feat_path)

        for sex_label, sex_code in [("Male", "M"), ("Female", "F")]:
            df_sex = df[df["sex"] == sex_code].copy()
            df_anthro = df_sex[df_sex["has_anthro"] == 1].copy()

            if len(df_anthro) < min_anthro:
                logger.info(
                    f"cutoff={cutoff} {sex_label}: anthro subsample {len(df_anthro)} "
                    f"< {min_anthro}, skipping"
                )
                continue

            avail_traj = [c for c in traj_cols if c in df_sex.columns]
            avail_anthro = [c for c in anthro_cols if c in df_sex.columns]

            y_full = df_sex["lifetime_pb"].values.astype(np.float64)
            y_sub = df_anthro["lifetime_pb"].values.astype(np.float64)
            X_full_traj = df_sex[avail_traj].values.astype(np.float64)
            X_sub_traj = df_anthro[avail_traj].values.astype(np.float64)
            X_sub_anthro = df_anthro[avail_anthro].values.astype(np.float64)

            models = get_models(config)
            for model_name, model in models.items():
                # Version 1: Full sample, no anthro
                r1 = random_cv(model, X_full_traj, y_full, n_folds, n_repeats, seed)

                # Version 2: Anthro subsample, no anthro features
                r2 = random_cv(model, X_sub_traj, y_sub, n_folds, n_repeats, seed)

                # Version 3: Anthro subsample, with anthro features
                r3 = random_cv(model, X_sub_anthro, y_sub, n_folds, n_repeats, seed)

                delta_r2 = r3["R2_mean"] - r2["R2_mean"]

                logger.info(
                    f"cutoff={cutoff} {sex_label} {model_name}: "
                    f"Full={r1['R2_mean']:.4f}  Sub(no)={r2['R2_mean']:.4f}  "
                    f"Sub(+anthro)={r3['R2_mean']:.4f}  ΔR²={delta_r2:+.4f}"
                )

                all_results.append({
                    "cutoff_age": cutoff, "sex": sex_label, "model": model_name,
                    "R2_full_no_anthro": r1["R2_mean"],
                    "R2_sub_no_anthro": r2["R2_mean"],
                    "R2_sub_with_anthro": r3["R2_mean"],
                    "delta_R2": delta_r2,
                    "n_full": r1["n_samples"],
                    "n_anthro_sub": r2["n_samples"],
                })

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(results_dir / "anthro_results.csv", index=False)
    df_results.to_csv(tables_dir / "table3_anthro_ablation.csv", index=False)
    logger.info(f"Saved {len(df_results)} anthro result rows")


if __name__ == "__main__":
    main()
