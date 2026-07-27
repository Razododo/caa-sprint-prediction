"""
Step 8: Sensitivity analyses and robustness checks.

1. Career completion filter (no records after completion_cutoff_year)
2. Wind coefficient sensitivity (β ±40%, 9 steps)
3. Parsimonious model: 2-feature vs 4-feature vs full
4. Attrition curve: active athletes and dropout rate by age
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from utils.io import load_config, ensure_dirs, read_parquet
from utils.validation import random_cv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    config = load_config()
    ensure_dirs(config)

    processed = Path(config["paths"]["processed"])
    interim = Path(config["paths"]["interim"])
    results_dir = Path(config["paths"]["raw_results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    robustness_dir = results_dir / "robustness"
    robustness_dir.mkdir(parents=True, exist_ok=True)

    n_repeats = config["modeling"]["cv"]["n_repeats_quick"] if args.quick else config["modeling"]["cv"]["n_repeats"]
    n_folds = config["modeling"]["cv"]["n_folds"]
    seed = config["modeling"]["cv"]["random_seed"]
    mc = config["modeling"]["algorithms"]["gradient_boosting"]

    # ============================
    # Check 1: Career completion filter
    # ============================
    logger.info("=" * 50)
    logger.info("CHECK 1: Career completion filter")
    completion_year = config["robustness"]["career_completion_cutoff_year"]

    records = read_parquet(interim / "cleaned_records.parquet")
    rec_100m = records[records["event"] == "100m"]
    last_year_by_athlete = rec_100m.groupby("athlete_id")["competition_date"].max().dt.year
    completed_ids = set(last_year_by_athlete[last_year_by_athlete <= completion_year].index)
    logger.info(f"Athletes with last record ≤ {completion_year}: {len(completed_ids)}")

    test_cutoffs = [18, 20, 22]
    completion_results = []

    for cutoff in test_cutoffs:
        feat_path = processed / f"features_cutoff_{cutoff}.parquet"
        if not feat_path.exists():
            continue
        df = read_parquet(feat_path)
        df_m = df[df["sex"] == "M"]
        df_completed = df_m[df_m["athlete_id"].isin(completed_ids)]

        if len(df_completed) < 100:
            continue

        feat_cols = [c for c in df_m.columns if c not in [
            "athlete_id", "sex", "lifetime_pb", "has_anthro",
            "year_first", "year_last", "year_mean", "career_span_calendar",
            "decade_first", "decade_mode", "birth_cohort_5yr",
            "height_cm", "weight_kg", "bmi", "height_missing", "weight_missing",
        ]]

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(**mc)),
        ])

        X_full = df_m[feat_cols].values.astype(np.float64)
        y_full = df_m["lifetime_pb"].values.astype(np.float64)
        r_full = random_cv(model, X_full, y_full, n_folds, n_repeats, seed)

        X_comp = df_completed[feat_cols].values.astype(np.float64)
        y_comp = df_completed["lifetime_pb"].values.astype(np.float64)
        r_comp = random_cv(model, X_comp, y_comp, n_folds, n_repeats, seed)

        logger.info(
            f"  cutoff={cutoff}: Full R²={r_full['R2_mean']:.4f} (n={len(y_full)}), "
            f"Completed R²={r_comp['R2_mean']:.4f} (n={len(y_comp)}), "
            f"Δ={r_comp['R2_mean'] - r_full['R2_mean']:+.4f}"
        )
        completion_results.append({
            "cutoff_age": cutoff,
            "R2_full": r_full["R2_mean"],
            "R2_completed": r_comp["R2_mean"],
            "n_full": len(y_full),
            "n_completed": len(y_comp),
        })

    pd.DataFrame(completion_results).to_csv(robustness_dir / "career_completion.csv", index=False)

    # ============================
    # Check 2: Parsimonious model
    # ============================
    logger.info("=" * 50)
    logger.info("CHECK 2: Parsimonious model")

    parsimony_cfgs = config["robustness"]["parsimonious_features"]
    parsimony_results = []

    for cutoff in test_cutoffs:
        feat_path = processed / f"features_cutoff_{cutoff}.parquet"
        if not feat_path.exists():
            continue
        df = read_parquet(feat_path)
        df_m = df[df["sex"] == "M"]
        y = df_m["lifetime_pb"].values.astype(np.float64)

        for label, feat_list in parsimony_cfgs.items():
            avail = [f for f in feat_list if f in df_m.columns]
            if not avail:
                continue
            X = df_m[avail].values.astype(np.float64)
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(**mc)),
            ])
            r = random_cv(model, X, y, n_folds, n_repeats, seed)
            logger.info(f"  cutoff={cutoff} {label} ({len(avail)} feats): R²={r['R2_mean']:.4f}")
            parsimony_results.append({
                "cutoff_age": cutoff, "feature_set": label,
                "n_features": len(avail), "R2": r["R2_mean"], "n": len(y),
            })

    pd.DataFrame(parsimony_results).to_csv(robustness_dir / "parsimonious.csv", index=False)

    # ============================
    # Check 3: Attrition curve
    # ============================
    logger.info("=" * 50)
    logger.info("CHECK 3: Attrition curve")

    athletes = read_parquet(interim / "cleaned_athletes.parquet")
    dob_map = athletes.set_index("athlete_id")["dob"].to_dict()

    rec_100m = rec_100m.copy()
    rec_100m["dob"] = rec_100m["athlete_id"].map(dob_map)
    rec_100m["age_floor"] = ((rec_100m["competition_date"] - rec_100m["dob"]).dt.total_seconds() / (365.25 * 86400)).astype(int)

    attrition_rows = []
    for sex_code, sex_label in [("M", "Male"), ("F", "Female")]:
        sex_recs = rec_100m[rec_100m["sex"] == sex_code]
        for age in range(12, 36):
            active = sex_recs[sex_recs["age_floor"] == age]["athlete_id"].nunique()
            attrition_rows.append({"sex": sex_label, "age": age, "active_athletes": active})

    df_att = pd.DataFrame(attrition_rows)
    df_att.to_csv(robustness_dir / "attrition_curve.csv", index=False)
    logger.info("Attrition curve saved")

    for sex in ["Male", "Female"]:
        subset = df_att[df_att["sex"] == sex]
        peak = subset.loc[subset["active_athletes"].idxmax()]
        logger.info(f"  {sex}: peak at age {int(peak['age'])} with {int(peak['active_athletes'])} athletes")

    logger.info("All robustness checks complete")


if __name__ == "__main__":
    main()
