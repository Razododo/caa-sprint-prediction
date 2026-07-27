"""
Step 12: Anthropometric subsample analysis — three-way comparison.

Version 1: Full Group A, no anthro features  (loaded from s11 results)
Version 2: Anthro subsample, no anthro features  (selection bias check)
Version 3: Anthro subsample, WITH anthro features  (anthro contribution)

ΔR² = V3 - V2  (NOT V3 - V1)
Selection bias = V2 - V1

Versions 2 and 3 use identical CV folds (same random seeds).

Input:  data/processed/group_labels.parquet
        data/interim/cleaned_records.parquet
        data/interim/cleaned_athletes.parquet
        results/raw_results/controlled_analysis/group_a_results.csv
Output: results/raw_results/anthro/anthro_results.csv
        results/tables/table3_anthro_ablation.csv
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import load_config, ensure_dirs, read_parquet
from utils.features import (
    compute_trajectory_raw,
    compute_trajectory_wc,
    compute_trajectory_dynamics,
    compute_career_structure,
    compute_round_performance,
    compute_wind_features,
    compute_era_features,
)
from utils.validation import random_cv

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def load_feature_registry() -> dict:
    project_root = Path(__file__).resolve().parent.parent
    return yaml.safe_load((project_root / "config" / "feature_registry.yaml").read_text())


TRAJ_WIND_GROUPS = [
    "trajectory_raw", "trajectory_wc", "trajectory_dynamics",
    "career_structure", "round_performance", "wind",
]


def get_group_columns(registry: dict, group_names: list[str]) -> list[str]:
    cols = []
    for g in group_names:
        cols.extend(f["name"] for f in registry["groups"][g]["features"])
    return cols


def build_features(
    athlete_ids: list[str],
    records_100m: pd.DataFrame,
    athletes_df: pd.DataFrame,
    cutoff_age: float,
    min_for_slope: int = 3,
) -> pd.DataFrame:
    """Build feature matrix from scratch for a set of athletes."""
    dob_map = athletes_df.set_index("athlete_id")["dob"].to_dict()

    pre = records_100m[
        (records_100m["athlete_id"].isin(athlete_ids))
        & (records_100m["age_at_comp"] <= cutoff_age)
    ]

    rows = []
    for aid, recs in pre.groupby("athlete_id"):
        feats = {}
        feats.update(compute_trajectory_raw(recs))
        feats.update(compute_trajectory_wc(recs))
        feats.update(compute_trajectory_dynamics(recs, min_for_slope))
        feats.update(compute_career_structure(recs))
        feats.update(compute_round_performance(recs))
        feats.update(compute_wind_features(recs))
        dob = dob_map.get(aid)
        if dob is not None:
            feats.update(compute_era_features(recs, dob))
        feats["athlete_id"] = aid
        rows.append(feats)

    return pd.DataFrame(rows).set_index("athlete_id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    config = load_config()
    ensure_dirs(config)
    registry = load_feature_registry()

    interim = Path(config["paths"]["interim"])
    processed = Path(config["paths"]["processed"])
    s11_dir = Path(config["paths"]["raw_results"]) / "controlled_analysis"
    out_dir = Path(config["paths"]["raw_results"]) / "anthro"
    tables_dir = Path(config["paths"]["tables"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    n_folds = config["modeling"]["cv"]["n_folds"]
    seed = config["modeling"]["cv"]["random_seed"]
    n_repeats = 20 if args.quick else 500
    min_for_slope = config["features"]["min_records_for_slope"]
    bmi_range = config["anthropometric"]["bmi_range"]
    mc = config["modeling"]["algorithms"]["gradient_boosting"]

    records = read_parquet(interim / "cleaned_records.parquet")
    athletes = read_parquet(interim / "cleaned_athletes.parquet")
    group_labels = read_parquet(processed / "group_labels.parquet")
    rec_100m = records[records["event"] == "100m"].copy()

    # Load Version 1 from s11 results
    s11_results = pd.read_csv(s11_dir / "group_a_results.csv")

    traj_wind_cols = get_group_columns(registry, TRAJ_WIND_GROUPS)

    cutoffs = [16, 17, 18]
    all_results = []

    for cutoff in cutoffs:
        col = f"cutoff_{cutoff}_group"

        for sex_code, sex_label in [("M", "Male"), ("F", "Female")]:
            logger.info(f"\n{'='*60}")
            logger.info(f"ANTHRO ANALYSIS: Cutoff {cutoff}, {sex_label}")
            logger.info("=" * 60)

            # --- Version 1: Load from s11 ---
            v1_row = s11_results[
                (s11_results["cutoff_age"] == cutoff)
                & (s11_results["sex"] == sex_label)
                & (s11_results["model"] == "GradientBoosting")
                & (s11_results["variant"] == "traj_wind")
                & (s11_results["cv_strategy"] == "random")
            ]
            if len(v1_row) == 0:
                logger.warning(f"  No s11 result for cutoff={cutoff} {sex_label}, skip")
                continue

            r2_v1 = v1_row.iloc[0]["R2_mean"]
            n_v1 = int(v1_row.iloc[0]["n_samples"])
            logger.info(f"  V1 (full Group A, no anthro): R²={r2_v1:.4f}, n={n_v1}")

            # --- Identify anthro subsample ---
            anthro_ids = group_labels[
                (group_labels[col] == "A")
                & (group_labels["sex"] == sex_code)
                & (group_labels["has_anthro"] == True)
            ]["athlete_id"].tolist()

            if len(anthro_ids) < 30:
                logger.warning(f"  Anthro subsample too small: {len(anthro_ids)}, skip")
                continue

            logger.info(f"  Anthro subsample: n={len(anthro_ids)}")

            # Build features from scratch for anthro subsample
            feats = build_features(anthro_ids, rec_100m, athletes, cutoff, min_for_slope)

            # Add target
            pb_map = group_labels.set_index("athlete_id")["lifetime_pb"]
            feats["lifetime_pb"] = pb_map.loc[feats.index]
            feats = feats.dropna(subset=["lifetime_pb"])

            # Add anthropometric features
            h_map = group_labels.set_index("athlete_id")["height_cm"]
            w_map = group_labels.set_index("athlete_id")["weight_kg"]
            feats["height_cm"] = h_map.loc[feats.index]
            feats["weight_kg"] = w_map.loc[feats.index]
            feats["bmi"] = feats["weight_kg"] / (feats["height_cm"] / 100) ** 2

            # Flag BMI outliers
            bmi_valid = (feats["bmi"] >= bmi_range[0]) & (feats["bmi"] <= bmi_range[1])
            n_bmi_outlier = (~bmi_valid).sum()
            if n_bmi_outlier > 0:
                logger.info(f"  BMI outliers (outside {bmi_range}): {n_bmi_outlier}")

            y = feats["lifetime_pb"].values.astype(np.float64)

            # traj_wind columns (no anthro)
            avail_tw = [c for c in traj_wind_cols if c in feats.columns]
            X_no_anthro = feats[avail_tw].values.astype(np.float64)

            # traj_wind + anthro columns
            anthro_feat_cols = ["height_cm", "weight_kg", "bmi"]
            avail_with = avail_tw + anthro_feat_cols
            X_with_anthro = feats[avail_with].values.astype(np.float64)

            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(**mc)),
            ])

            # --- Version 2: Anthro subsample, NO anthro features ---
            # Same seed as Version 3 for identical folds
            r_v2 = random_cv(model, X_no_anthro, y, n_folds, n_repeats, seed)
            r2_v2 = r_v2["R2_mean"]
            logger.info(
                f"  V2 (anthro sub, no anthro feat): R²={r2_v2:.4f} "
                f"[{r_v2['R2_ci_lo']:.4f},{r_v2['R2_ci_hi']:.4f}], n={r_v2['n_samples']}"
            )

            # --- Version 3: Anthro subsample, WITH anthro features ---
            # Same seed → identical folds
            r_v3 = random_cv(model, X_with_anthro, y, n_folds, n_repeats, seed)
            r2_v3 = r_v3["R2_mean"]
            logger.info(
                f"  V3 (anthro sub, WITH anthro feat): R²={r2_v3:.4f} "
                f"[{r_v3['R2_ci_lo']:.4f},{r_v3['R2_ci_hi']:.4f}], n={r_v3['n_samples']}"
            )

            # --- Derived metrics ---
            selection_bias = r2_v2 - r2_v1
            anthro_delta = r2_v3 - r2_v2

            logger.info(f"  Selection bias (V2-V1):      {selection_bias:+.4f}")
            logger.info(f"  Anthro contribution (V3-V2): {anthro_delta:+.4f}")

            all_results.append({
                "cutoff_age": cutoff, "sex": sex_label,
                "n_full_A": n_v1,
                "n_anthro_sub": r_v2["n_samples"],
                "R2_v1_full_no_anthro": round(r2_v1, 4),
                "R2_v2_sub_no_anthro": round(r2_v2, 4),
                "R2_v2_ci_lo": round(r_v2["R2_ci_lo"], 4),
                "R2_v2_ci_hi": round(r_v2["R2_ci_hi"], 4),
                "R2_v3_sub_with_anthro": round(r2_v3, 4),
                "R2_v3_ci_lo": round(r_v3["R2_ci_lo"], 4),
                "R2_v3_ci_hi": round(r_v3["R2_ci_hi"], 4),
                "selection_bias": round(selection_bias, 4),
                "anthro_delta_R2": round(anthro_delta, 4),
                "RMSE_v2": round(r_v2["RMSE_mean"], 4),
                "RMSE_v3": round(r_v3["RMSE_mean"], 4),
            })

            # --- BMI-only vs height+weight ---
            bmi_only_cols = avail_tw + ["bmi"]
            hw_only_cols = avail_tw + ["height_cm", "weight_kg"]

            X_bmi = feats[bmi_only_cols].values.astype(np.float64)
            X_hw = feats[hw_only_cols].values.astype(np.float64)

            r_bmi = random_cv(model, X_bmi, y, n_folds, n_repeats, seed)
            r_hw = random_cv(model, X_hw, y, n_folds, n_repeats, seed)

            logger.info(
                f"  BMI only:  R²={r_bmi['R2_mean']:.4f}, "
                f"H+W only: R²={r_hw['R2_mean']:.4f}"
            )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "anthro_results.csv", index=False)
    df.to_csv(tables_dir / "table3_anthro_ablation.csv", index=False)
    logger.info(f"\nSaved anthro_results.csv ({len(df)} rows)")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("ANTHRO THREE-WAY SUMMARY")
    logger.info("=" * 70)
    logger.info(
        f"{'Cut':>3}  {'Sex':>6}  {'n_A':>5}  {'n_sub':>5}  "
        f"{'V1':>6}  {'V2':>6}  {'V3':>6}  {'bias':>6}  {'ΔR²':>6}"
    )
    for _, r in df.iterrows():
        logger.info(
            f"{int(r['cutoff_age']):>3}  {r['sex']:>6}  "
            f"{int(r['n_full_A']):>5}  {int(r['n_anthro_sub']):>5}  "
            f"{r['R2_v1_full_no_anthro']:>6.4f}  "
            f"{r['R2_v2_sub_no_anthro']:>6.4f}  "
            f"{r['R2_v3_sub_with_anthro']:>6.4f}  "
            f"{r['selection_bias']:>+6.4f}  "
            f"{r['anthro_delta_R2']:>+6.4f}"
        )


if __name__ == "__main__":
    main()
