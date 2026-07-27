"""
Step 11: Controlled developmental analysis — Group A only.

Core analysis producing Table 2, Fig 3, Fig 6, Fig 7 data.
Builds features FROM SCRATCH for Group A athletes at each cutoff.

Input:  data/processed/group_labels.parquet
        data/interim/cleaned_records.parquet
        data/interim/cleaned_athletes.parquet
Output: results/raw_results/controlled_analysis/group_a_results.csv
        results/raw_results/controlled_analysis/pb_stratified_results.csv
        results/tables/table2_group_a_performance.csv
"""
import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.base import clone

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
from sklearn.model_selection import KFold
from utils.validation import random_cv, province_disjoint_cv

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def load_feature_registry() -> dict:
    project_root = Path(__file__).resolve().parent.parent
    return yaml.safe_load((project_root / "config" / "feature_registry.yaml").read_text())


def get_variant_columns(registry: dict, variant: str) -> list[str]:
    """Get feature column names for a dataset variant."""
    vdef = registry["dataset_variants"][variant]
    cols = []
    for group_name in vdef["groups"]:
        group = registry["groups"][group_name]
        cols.extend(f["name"] for f in group["features"])
    return cols


def build_features_for_group_a(
    athlete_ids: list[str],
    records_100m: pd.DataFrame,
    athletes_df: pd.DataFrame,
    cutoff_age: float,
    min_for_slope: int = 3,
) -> pd.DataFrame:
    """
    Build feature matrix FROM SCRATCH for a specific set of Group A athletes.

    CRITICAL: Uses only pre-cutoff records. Does NOT subset a pre-built matrix.
    """
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


def get_models(config: dict) -> dict[str, Pipeline]:
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


# full_no_anthro = traj_wind + era (not in registry, define here)
VARIANT_GROUPS = {
    "trajectory_only": [
        "trajectory_raw", "trajectory_dynamics", "career_structure", "round_performance"
    ],
    "traj_wind": [
        "trajectory_raw", "trajectory_wc", "trajectory_dynamics",
        "career_structure", "round_performance", "wind"
    ],
    "full_no_anthro": [
        "trajectory_raw", "trajectory_wc", "trajectory_dynamics",
        "career_structure", "round_performance", "wind", "era"
    ],
}


def get_variant_cols(registry: dict, variant: str) -> list[str]:
    groups = VARIANT_GROUPS[variant]
    cols = []
    for g in groups:
        cols.extend(f["name"] for f in registry["groups"][g]["features"])
    return cols


def get_feature_to_group(registry: dict, variant: str) -> dict[str, str]:
    """Map each feature name to its group for aggregation."""
    out = {}
    for group_name in VARIANT_GROUPS[variant]:
        for f in registry["groups"][group_name]["features"]:
            out[f["name"]] = group_name
    return out


def run_permutation_importance_cv(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    feature_to_group: dict[str, str],
    n_folds: int = 5,
    n_repeats: int = 10,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Run CV; in each fold fit on train and compute permutation importance on test.
    Returns (per_feature_rows, per_group_rows).
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_importances = []  # list of (feature_names, importance_mean array)

    for train_idx, test_idx in kf.split(X):
        m = clone(model)
        m.fit(X[train_idx], y[train_idx])
        X_test = X[test_idx]
        y_test = y[test_idx]
        perm = permutation_importance(
            m, X_test, y_test, n_repeats=n_repeats, random_state=seed, scoring="r2"
        )
        fold_importances.append((perm.importances_mean, perm.importances_std))

    # Average across folds
    imp_mean = np.mean([f[0] for f in fold_importances], axis=0)
    imp_std = np.mean([f[1] for f in fold_importances], axis=0)

    per_feature = []
    for i, name in enumerate(feature_names):
        per_feature.append({
            "feature": name,
            "group": feature_to_group.get(name, "other"),
            "importance_mean": imp_mean[i],
            "importance_std": imp_std[i],
        })

    # Aggregate by group (mean of importances within group)
    by_group = defaultdict(list)
    for row in per_feature:
        by_group[row["group"]].append(row["importance_mean"])
    per_group = []
    for group_name, vals in sorted(by_group.items()):
        per_group.append({
            "feature_group": group_name,
            "r2_drop_mean": float(np.mean(vals)),
            "r2_drop_std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            "n_features": len(vals),
        })
    return per_feature, per_group


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    config = load_config()
    ensure_dirs(config)
    registry = load_feature_registry()

    interim = Path(config["paths"]["interim"])
    out_dir = Path(config["paths"]["raw_results"]) / "controlled_analysis"
    tables_dir = Path(config["paths"]["tables"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    n_folds = config["modeling"]["cv"]["n_folds"]
    seed = config["modeling"]["cv"]["random_seed"]
    n_jobs = config["modeling"]["cv"].get("n_jobs", 1)
    n_repeats_primary = 20 if args.quick else 500
    n_repeats_supp = 10 if args.quick else 100

    records = read_parquet(interim / "cleaned_records.parquet")
    athletes = read_parquet(interim / "cleaned_athletes.parquet")
    group_labels = read_parquet(Path(config["paths"]["processed"]) / "group_labels.parquet")

    rec_100m = records[records["event"] == "100m"].copy()

    primary_cutoffs = [16, 17, 18]
    supplementary_cutoffs = [19, 20, 21, 22]
    all_cutoffs = primary_cutoffs + supplementary_cutoffs
    variants = ["trajectory_only", "traj_wind", "full_no_anthro"]

    all_results = []

    for cutoff in all_cutoffs:
        is_primary = cutoff in primary_cutoffs
        n_rep = n_repeats_primary if is_primary else n_repeats_supp
        tag = "primary" if is_primary else "supplementary"

        for sex_code, sex_label in [("M", "Male"), ("F", "Female")]:
            col = f"cutoff_{cutoff}_group"
            a_ids = group_labels[
                (group_labels[col] == "A") & (group_labels["sex"] == sex_code)
            ]["athlete_id"].tolist()

            if len(a_ids) < 30:
                logger.warning(f"cutoff={cutoff} {sex_label}: only {len(a_ids)} Group A, skip")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Cutoff {cutoff} {sex_label} (n={len(a_ids)}, {tag}, {n_rep} reps)")
            logger.info("=" * 60)

            # Build features FROM SCRATCH
            feats = build_features_for_group_a(
                a_ids, rec_100m, athletes, cutoff,
                config["features"]["min_records_for_slope"],
            )

            # Add target
            pb_map = group_labels.set_index("athlete_id")["lifetime_pb"]
            feats["lifetime_pb"] = pb_map.loc[feats.index]
            feats = feats.dropna(subset=["lifetime_pb"])

            # Province for disjoint CV
            prov_map = group_labels.set_index("athlete_id")["province"]
            feats["province"] = prov_map.loc[feats.index]

            y = feats["lifetime_pb"].values.astype(np.float64)

            models = get_models(config)

            for variant in variants:
                feat_cols = get_variant_cols(registry, variant)
                avail = [c for c in feat_cols if c in feats.columns]
                if not avail:
                    continue

                X = feats[avail].values.astype(np.float64)

                for model_name, model in models.items():
                    # --- Random CV ---
                    r = random_cv(model, X, y, n_folds, n_rep, seed, n_jobs=n_jobs)
                    logger.info(
                        f"  {variant:20s} {model_name:18s} "
                        f"R²={r['R2_mean']:.4f} [{r['R2_ci_lo']:.4f},{r['R2_ci_hi']:.4f}] "
                        f"RMSE={r['RMSE_mean']:.4f}"
                    )
                    all_results.append({
                        "cutoff_age": cutoff, "sex": sex_label, "variant": variant,
                        "model": model_name, "cv_strategy": "random",
                        "n_samples": r["n_samples"], "n_features": len(avail),
                        "n_repeats": n_rep,
                        "R2_mean": r["R2_mean"], "R2_ci_lo": r["R2_ci_lo"],
                        "R2_ci_hi": r["R2_ci_hi"], "R2_std": r["R2_std"],
                        "RMSE_mean": r["RMSE_mean"], "MAE_mean": r["MAE_mean"],
                        "tag": tag,
                    })

                    # --- Province-disjoint CV (primary cutoffs, GBM only) ---
                    if is_primary and model_name == "GradientBoosting":
                        prov_vals = feats["province"].values
                        valid_prov = pd.notna(prov_vals)
                        if valid_prov.sum() >= 50:
                            prov_groups = prov_vals[valid_prov]
                            unique_provs = pd.Series(prov_groups).value_counts()
                            ok_provs = unique_provs[unique_provs >= 5].index
                            mask = valid_prov & pd.Series(prov_vals).isin(ok_provs).values
                            if mask.sum() >= 50 and len(ok_provs) >= 5:
                                rp = province_disjoint_cv(
                                    model, X[mask], y[mask], prov_vals[mask],
                                    n_folds=min(5, len(ok_provs)),
                                )
                                if rp:
                                    logger.info(
                                        f"  {'':20s} {'Prov-disjoint':18s} "
                                        f"R²={rp['R2_mean']:.4f} "
                                        f"({rp['n_groups']} provinces)"
                                    )
                                    all_results.append({
                                        "cutoff_age": cutoff, "sex": sex_label,
                                        "variant": variant, "model": model_name,
                                        "cv_strategy": "province_disjoint",
                                        "n_samples": rp["n_samples"],
                                        "n_features": len(avail),
                                        "n_repeats": None,
                                        "R2_mean": rp["R2_mean"],
                                        "R2_std": rp["R2_std"],
                                        "RMSE_mean": rp["RMSE_mean"],
                                        "MAE_mean": rp["MAE_mean"],
                                        "n_groups": rp.get("n_groups"),
                                        "tag": tag,
                                    })

    df_results = pd.DataFrame(all_results)

    # ------------------------------------------------------------------
    # PB-stratified analysis within Group A
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("PB-STRATIFIED ANALYSIS (Group A)")
    logger.info("=" * 70)

    pb_strata_male = [
        ("elite", 0, 11.0),
        ("sub-elite", 11.0, 12.0),
        ("grassroots", 12.0, 99.0),
    ]
    pb_strata_female = [
        ("elite", 0, 12.5),
        ("sub-elite", 12.5, 13.5),
        ("grassroots", 13.5, 99.0),
    ]

    strat_results = []

    for cutoff in primary_cutoffs:
        for sex_code, sex_label, strata in [
            ("M", "Male", pb_strata_male),
            ("F", "Female", pb_strata_female),
        ]:
            col = f"cutoff_{cutoff}_group"
            a_mask = (group_labels[col] == "A") & (group_labels["sex"] == sex_code)
            a_ids = group_labels.loc[a_mask, "athlete_id"].tolist()

            if len(a_ids) < 30:
                continue

            feats = build_features_for_group_a(
                a_ids, rec_100m, athletes, cutoff,
                config["features"]["min_records_for_slope"],
            )
            feats["lifetime_pb"] = pb_map.loc[feats.index]
            feats = feats.dropna(subset=["lifetime_pb"])

            variant = "traj_wind"
            feat_cols = get_variant_cols(registry, variant)
            avail = [c for c in feat_cols if c in feats.columns]

            for stratum_name, pb_lo, pb_hi in strata:
                mask = (feats["lifetime_pb"] >= pb_lo) & (feats["lifetime_pb"] < pb_hi)
                sub = feats[mask]
                if len(sub) < 30:
                    logger.info(
                        f"  cutoff={cutoff} {sex_label} {stratum_name}: "
                        f"n={len(sub)} (skip, <30)"
                    )
                    strat_results.append({
                        "cutoff_age": cutoff, "sex": sex_label,
                        "stratum": stratum_name, "pb_range": f"{pb_lo}-{pb_hi}",
                        "n": len(sub), "pb_sd": sub["lifetime_pb"].std() if len(sub) > 1 else np.nan,
                        "R2_mean": np.nan, "note": "insufficient sample",
                    })
                    continue

                X_s = sub[avail].values.astype(np.float64)
                y_s = sub["lifetime_pb"].values.astype(np.float64)

                model = get_models(config)["GradientBoosting"]
                r = random_cv(model, X_s, y_s, n_folds, n_repeats_primary, seed, n_jobs=n_jobs)
                logger.info(
                    f"  cutoff={cutoff} {sex_label} {stratum_name}: "
                    f"n={len(sub)}, PB_SD={sub['lifetime_pb'].std():.3f}, "
                    f"R²={r['R2_mean']:.4f} [{r['R2_ci_lo']:.4f},{r['R2_ci_hi']:.4f}]"
                )
                strat_results.append({
                    "cutoff_age": cutoff, "sex": sex_label,
                    "stratum": stratum_name, "pb_range": f"{pb_lo}-{pb_hi}",
                    "n": len(sub),
                    "pb_sd": round(sub["lifetime_pb"].std(), 4),
                    "pb_mean": round(sub["lifetime_pb"].mean(), 3),
                    "R2_mean": r["R2_mean"], "R2_ci_lo": r.get("R2_ci_lo"),
                    "R2_ci_hi": r.get("R2_ci_hi"),
                    "RMSE_mean": r["RMSE_mean"],
                })

    # ------------------------------------------------------------------
    # Permutation importance (cutoff 18, Group A, GradientBoosting full_no_anthro)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("PERMUTATION IMPORTANCE (cutoff 18, Group A, GBM full_no_anthro)")
    logger.info("=" * 70)

    perm_group_rows = []
    perm_feature_rows = []
    feat_to_group = get_feature_to_group(registry, "full_no_anthro")
    gb_model = get_models(config)["GradientBoosting"]
    n_perm_repeats = 10
    n_perm_folds = 5

    for sex_code, sex_label in [("M", "Male"), ("F", "Female")]:
        col = "cutoff_18_group"
        a_ids = group_labels[
            (group_labels[col] == "A") & (group_labels["sex"] == sex_code)
        ]["athlete_id"].tolist()
        if len(a_ids) < 50:
            logger.warning(f"  {sex_label}: n={len(a_ids)} Group A, skip permutation importance")
            continue

        feats = build_features_for_group_a(
            a_ids, rec_100m, athletes, 18,
            config["features"]["min_records_for_slope"],
        )
        feats["lifetime_pb"] = pb_map.loc[feats.index]
        feats = feats.dropna(subset=["lifetime_pb"])
        feat_cols = get_variant_cols(registry, "full_no_anthro")
        avail = [c for c in feat_cols if c in feats.columns]
        X = feats[avail].values.astype(np.float64)
        y = feats["lifetime_pb"].values.astype(np.float64)

        per_f, per_g = run_permutation_importance_cv(
            gb_model, X, y, avail, feat_to_group,
            n_folds=n_perm_folds, n_repeats=n_perm_repeats, seed=seed,
        )
        for row in per_g:
            perm_group_rows.append({"sex": sex_label, **row})
        for row in per_f:
            perm_feature_rows.append({"sex": sex_label, **row})

        ranks = sorted(per_g, key=lambda x: -x["r2_drop_mean"])
        logger.info(f"  {sex_label} (n={len(y)}): top groups " + ", ".join(
            f"{r['feature_group']}={r['r2_drop_mean']:.4f}" for r in ranks[:5]
        ))

    # ------------------------------------------------------------------
    # Group A parsimonious (cutoff 18, 2-feature: best_time + slope)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("GROUP A PARSIMONIOUS (cutoff 18, 2-feature: best_time_raw + improvement_slope_raw)")
    logger.info("=" * 70)

    parsimonious_2_cols = ["best_time_raw", "improvement_slope_raw"]
    group_a_parsimonious_rows = []

    for sex_code, sex_label in [("M", "Male"), ("F", "Female")]:
        col = "cutoff_18_group"
        a_ids = group_labels[
            (group_labels[col] == "A") & (group_labels["sex"] == sex_code)
        ]["athlete_id"].tolist()
        if len(a_ids) < 30:
            continue

        feats = build_features_for_group_a(
            a_ids, rec_100m, athletes, 18,
            config["features"]["min_records_for_slope"],
        )
        feats["lifetime_pb"] = pb_map.loc[feats.index]
        feats = feats.dropna(subset=["lifetime_pb"])
        avail = [c for c in parsimonious_2_cols if c in feats.columns]
        if len(avail) < 2:
            logger.warning(f"  {sex_label}: missing cols {parsimonious_2_cols}")
            continue

        X_pars = feats[avail].values.astype(np.float64)
        y_pars = feats["lifetime_pb"].values.astype(np.float64)
        r_pars = random_cv(gb_model, X_pars, y_pars, n_folds, n_repeats_primary, seed, n_jobs=n_jobs)
        full_row = df_results[
            (df_results["cutoff_age"] == 18)
            & (df_results["sex"] == sex_label)
            & (df_results["variant"] == "full_no_anthro")
            & (df_results["model"] == "GradientBoosting")
            & (df_results["cv_strategy"] == "random")
        ]
        r2_full = float(full_row["R2_mean"].iloc[0]) if len(full_row) else np.nan
        r2_pars = r_pars["R2_mean"]
        retention = r2_pars / r2_full if r2_full and r2_full > 0 else np.nan
        logger.info(
            f"  {sex_label}: R²_2feat={r2_pars:.4f}, R²_full={r2_full:.4f}, "
            f"retention={retention:.2%} (n={len(y_pars)})"
        )
        group_a_parsimonious_rows.append({
            "cutoff_age": 18,
            "sex": sex_label,
            "n": len(y_pars),
            "R2_parsimonious_2feat": round(r2_pars, 4),
            "R2_full_no_anthro": round(r2_full, 4),
            "retention_pct": round(100 * retention, 2) if not np.isnan(retention) else None,
        })

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    df_results.to_csv(out_dir / "group_a_results.csv", index=False)
    logger.info(f"\nSaved group_a_results.csv ({len(df_results)} rows)")

    if perm_group_rows:
        pd.DataFrame(perm_group_rows).to_csv(
            out_dir / "permutation_importance_by_group.csv", index=False
        )
        logger.info(f"Saved permutation_importance_by_group.csv ({len(perm_group_rows)} rows)")
    if perm_feature_rows:
        pd.DataFrame(perm_feature_rows).to_csv(
            out_dir / "permutation_importance_by_feature.csv", index=False
        )
        logger.info(f"Saved permutation_importance_by_feature.csv ({len(perm_feature_rows)} rows)")
    if group_a_parsimonious_rows:
        pd.DataFrame(group_a_parsimonious_rows).to_csv(
            out_dir / "group_a_parsimonious_cutoff18.csv", index=False
        )
        logger.info(f"Saved group_a_parsimonious_cutoff18.csv ({len(group_a_parsimonious_rows)} rows)")

    df_strat = pd.DataFrame(strat_results)
    df_strat.to_csv(out_dir / "pb_stratified_results.csv", index=False)
    logger.info(f"Saved pb_stratified_results.csv ({len(df_strat)} rows)")

    # Table 2: primary results only, random CV
    primary_random = df_results[
        (df_results["tag"] == "primary") & (df_results["cv_strategy"] == "random")
    ]
    primary_random.to_csv(tables_dir / "table2_group_a_performance.csv", index=False)
    logger.info(f"Saved table2_group_a_performance.csv")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: GradientBoosting, traj_wind, Random CV")
    logger.info("=" * 70)
    gb_tw = df_results[
        (df_results["model"] == "GradientBoosting")
        & (df_results["variant"] == "traj_wind")
        & (df_results["cv_strategy"] == "random")
    ]
    for _, r in gb_tw.iterrows():
        logger.info(
            f"  cutoff={int(r['cutoff_age']):>2} {r['sex']:>6} "
            f"R²={r['R2_mean']:.4f} [{r.get('R2_ci_lo',''):>6}] "
            f"n={int(r['n_samples'])}"
        )


if __name__ == "__main__":
    main()
