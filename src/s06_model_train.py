"""
Step 6: Train models with 3 CV strategies.

Input:  data/processed/features_cutoff_{age}.parquet
Output: results/raw_results/model_results.csv (all results in one table)
        results/tables/table2_model_performance.csv

For each {sex, cutoff_age, dataset_variant}:
    For each model (Ridge, RF, GB):
        1. Random 5-fold CV (n_repeats) → R², RMSE, MAE with 95% CI
        2. Era-disjoint CV → R² per split

Uses functions from utils/validation.py.
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
from utils.validation import random_cv, era_disjoint_cv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_feature_registry() -> dict:
    project_root = Path(__file__).resolve().parent.parent
    return yaml.safe_load((project_root / "config" / "feature_registry.yaml").read_text())


def get_feature_columns(registry: dict, variant: str) -> list[str]:
    """Get the list of feature column names for a dataset variant."""
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
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(**mc["random_forest"])),
        ]),
        "GradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(**mc["gradient_boosting"])),
        ]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use fewer CV repeats")
    args = parser.parse_args()

    config = load_config()
    ensure_dirs(config)
    registry = load_feature_registry()

    cutoff_ages = config["features"]["cutoff_ages"]
    n_repeats = config["modeling"]["cv"]["n_repeats_quick"] if args.quick else config["modeling"]["cv"]["n_repeats"]
    n_folds = config["modeling"]["cv"]["n_folds"]
    seed = config["modeling"]["cv"]["random_seed"]
    era_splits = config["modeling"]["era_splits"]
    processed = Path(config["paths"]["processed"])
    results_dir = Path(config["paths"]["raw_results"])
    tables_dir = Path(config["paths"]["tables"])
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Main analysis uses trajectory_only and traj_wind (full sample, no anthro)
    variants_main = ["trajectory_only", "traj_wind"]

    all_results = []

    for cutoff in cutoff_ages:
        feat_path = processed / f"features_cutoff_{cutoff}.parquet"
        if not feat_path.exists():
            logger.warning(f"Missing {feat_path}, skipping cutoff {cutoff}")
            continue

        df = read_parquet(feat_path)

        for sex_label, sex_code in [("Male", "M"), ("Female", "F")]:
            df_sex = df[df["sex"] == sex_code].copy()
            if len(df_sex) < 50:
                logger.warning(f"Cutoff {cutoff}, {sex_label}: only {len(df_sex)} athletes, skipping")
                continue

            for variant in variants_main:
                feat_cols = get_feature_columns(registry, variant)
                available = [c for c in feat_cols if c in df_sex.columns]
                if not available:
                    continue

                X = df_sex[available].values.astype(np.float64)
                y = df_sex["lifetime_pb"].values.astype(np.float64)

                valid = np.isfinite(y)
                X, y = X[valid], y[valid]
                if len(y) < 50:
                    continue

                models = get_models(config)
                for model_name, model in models.items():
                    logger.info(
                        f"cutoff={cutoff} sex={sex_label} variant={variant} "
                        f"model={model_name} n={len(y)} repeats={n_repeats}"
                    )

                    # Random CV
                    cv_result = random_cv(
                        model, X, y,
                        n_folds=n_folds,
                        n_repeats=n_repeats,
                        random_seed=seed,
                    )
                    row = {
                        "cutoff_age": cutoff,
                        "sex": sex_label,
                        "variant": variant,
                        "model": model_name,
                        "cv_strategy": "random",
                        **cv_result,
                    }
                    all_results.append(row)
                    logger.info(
                        f"  → R²={cv_result['R2_mean']:.4f} "
                        f"[{cv_result['R2_ci_lo']:.4f}, {cv_result['R2_ci_hi']:.4f}]"
                    )

                    # Era-disjoint CV
                    year_last = df_sex.loc[valid, "year_last"].values if "year_last" in df_sex.columns else None
                    if year_last is not None:
                        era_results = era_disjoint_cv(model, X, y, year_last, era_splits)
                        for er in era_results:
                            era_row = {
                                "cutoff_age": cutoff,
                                "sex": sex_label,
                                "variant": variant,
                                "model": model_name,
                                "cv_strategy": f"era_{er['split_label']}",
                                "R2_mean": er["R2"],
                                "RMSE_mean": er["RMSE"],
                                "MAE_mean": er["MAE"],
                                "n_samples": er["n_train"] + er["n_test"],
                            }
                            all_results.append(era_row)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(results_dir / "model_results.csv", index=False)
    logger.info(f"Saved {len(df_results)} result rows to model_results.csv")

    # Generate Table 2: best model per {sex, cutoff, variant} for random CV
    random_only = df_results[df_results["cv_strategy"] == "random"].copy()
    if len(random_only) > 0:
        random_only.to_csv(tables_dir / "table2_model_performance.csv", index=False)
        logger.info(f"Saved table2_model_performance.csv")

        # Summary: R² by cutoff for best model (GradientBoosting, traj_wind)
        logger.info("\n" + "=" * 60)
        logger.info("R² SUMMARY (GradientBoosting, traj_wind, random CV)")
        logger.info("=" * 60)
        gb_wind = random_only[
            (random_only["model"] == "GradientBoosting") &
            (random_only["variant"] == "traj_wind")
        ]
        for _, r in gb_wind.iterrows():
            logger.info(
                f"  cutoff={int(r['cutoff_age']):2d}  {r['sex']:6s}  "
                f"R²={r['R2_mean']:.4f} [{r.get('R2_ci_lo', 0):.4f}, {r.get('R2_ci_hi', 0):.4f}]  "
                f"RMSE={r['RMSE_mean']:.4f}  n={int(r['n_samples'])}"
            )


if __name__ == "__main__":
    main()
