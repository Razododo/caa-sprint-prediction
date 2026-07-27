"""
One-off script: run only permutation importance and Group A parsimonious at cutoff 18.
Uses same logic as s11. Outputs:
  results/raw_results/controlled_analysis/permutation_importance_by_group.csv
  results/raw_results/controlled_analysis/permutation_importance_by_feature.csv
  results/raw_results/controlled_analysis/group_a_parsimonious_cutoff18.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from sklearn.base import clone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from utils.io import load_config, read_parquet
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

VARIANT_GROUPS = {
    "full_no_anthro": [
        "trajectory_raw", "trajectory_wc", "trajectory_dynamics",
        "career_structure", "round_performance", "wind", "era"
    ],
}


def load_registry():
    with open(ROOT / "config" / "feature_registry.yaml") as f:
        return yaml.safe_load(f)


def get_variant_cols(registry, variant):
    groups = VARIANT_GROUPS[variant]
    cols = []
    for g in groups:
        cols.extend(f["name"] for f in registry["groups"][g]["features"])
    return cols


def get_feature_to_group(registry, variant):
    out = {}
    for group_name in VARIANT_GROUPS[variant]:
        for f in registry["groups"][group_name]["features"]:
            out[f["name"]] = group_name
    return out


def build_features_for_group_a(athlete_ids, records_100m, athletes_df, cutoff_age, min_for_slope=3):
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


def run_permutation_importance_cv(model, X, y, feature_names, feature_to_group, n_folds=5, n_repeats=10, seed=42):
    from collections import defaultdict
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_importances = []
    for train_idx, test_idx in kf.split(X):
        m = clone(model)
        m.fit(X[train_idx], y[train_idx])
        perm = permutation_importance(
            m, X[test_idx], y[test_idx], n_repeats=n_repeats, random_state=seed, scoring="r2"
        )
        fold_importances.append((perm.importances_mean, perm.importances_std))
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
    by_group = defaultdict(list)
    for row in per_feature:
        by_group[row["group"]].append(row["importance_mean"])
    per_group = []
    for group_name in sorted(by_group.keys()):
        vals = by_group[group_name]
        per_group.append({
            "feature_group": group_name,
            "r2_drop_mean": float(np.mean(vals)),
            "r2_drop_std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            "n_features": len(vals),
        })
    return per_feature, per_group


def main():
    config = load_config()
    registry = load_registry()
    interim = Path(config["paths"]["interim"])
    processed = Path(config["paths"]["processed"])
    out_dir = Path(config["paths"]["raw_results"]) / "controlled_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_parquet(interim / "cleaned_records.parquet")
    athletes = read_parquet(interim / "cleaned_athletes.parquet")
    group_labels = read_parquet(processed / "group_labels.parquet")
    rec_100m = records[records["event"] == "100m"].copy()
    pb_map = group_labels.set_index("athlete_id")["lifetime_pb"]
    min_slope = config["features"]["min_records_for_slope"]
    n_folds = config["modeling"]["cv"]["n_folds"]
    seed = config["modeling"]["cv"]["random_seed"]
    mc = config["modeling"]["algorithms"]["gradient_boosting"]
    n_repeats = 50  # fewer for quick run

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(**mc)),
    ])
    feat_to_group = get_feature_to_group(registry, "full_no_anthro")
    perm_group_rows = []
    perm_feature_rows = []
    parsimonious_rows = []
    parsimonious_2_cols = ["best_time_raw", "improvement_slope_raw"]

    for sex_code, sex_label in [("M", "Male"), ("F", "Female")]:
        a_ids = group_labels[
            (group_labels["cutoff_18_group"] == "A") & (group_labels["sex"] == sex_code)
        ]["athlete_id"].tolist()
        if len(a_ids) < 50:
            continue

        feats = build_features_for_group_a(a_ids, rec_100m, athletes, 18, min_slope)
        feats["lifetime_pb"] = pb_map.loc[feats.index]
        feats = feats.dropna(subset=["lifetime_pb"])
        y = feats["lifetime_pb"].values.astype(np.float64)

        # Permutation importance (full_no_anthro)
        feat_cols = get_variant_cols(registry, "full_no_anthro")
        avail = [c for c in feat_cols if c in feats.columns]
        X = feats[avail].values.astype(np.float64)
        per_f, per_g = run_permutation_importance_cv(
            model, X, y, avail, feat_to_group, n_folds=5, n_repeats=10, seed=seed
        )
        for row in per_g:
            perm_group_rows.append({"sex": sex_label, **row})
        for row in per_f:
            perm_feature_rows.append({"sex": sex_label, **row})
        print(f"Permutation importance {sex_label}: top groups",
              [(r["feature_group"], f"{r['r2_drop_mean']:.4f}") for r in sorted(per_g, key=lambda x: -x["r2_drop_mean"])[:5]])

        # Parsimonious 2-feature
        avail2 = [c for c in parsimonious_2_cols if c in feats.columns]
        if len(avail2) >= 2:
            X_pars = feats[avail2].values.astype(np.float64)
            r_pars = random_cv(model, X_pars, y, n_folds, n_repeats, seed)
            # Full model R² from same sample (we run it here for consistency)
            r_full = random_cv(model, X, y, n_folds, n_repeats, seed)
            r2_full = r_full["R2_mean"]
            r2_pars = r_pars["R2_mean"]
            retention = r2_pars / r2_full if r2_full > 0 else np.nan
            parsimonious_rows.append({
                "cutoff_age": 18,
                "sex": sex_label,
                "n": len(y),
                "R2_parsimonious_2feat": round(r2_pars, 4),
                "R2_full_no_anthro": round(r2_full, 4),
                "retention_pct": round(100 * retention, 2) if not np.isnan(retention) else None,
            })
            print(f"Parsimonious {sex_label}: R²_2feat={r2_pars:.4f}, R²_full={r2_full:.4f}, retention={retention:.2%}")

    pd.DataFrame(perm_group_rows).to_csv(out_dir / "permutation_importance_by_group.csv", index=False)
    pd.DataFrame(perm_feature_rows).to_csv(out_dir / "permutation_importance_by_feature.csv", index=False)
    pd.DataFrame(parsimonious_rows).to_csv(out_dir / "group_a_parsimonious_cutoff18.csv", index=False)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
