# -*- coding: utf-8 -*-
"""
Recompute Supplementary Table S4 (comparison of boosting frameworks) and
Supplementary Table S13 (province groups entering GroupKFold).

Usage (from the project root):
    python recompute_s4_s13.py

S13 needs no additional dependencies and is computed first. S4 requires xgboost
and lightgbm; if either is missing the script prints the install command and
skips S4 without interrupting the S13 output.

S4 uses the same protocol as Main Table 2: 500 repeats of 5-fold cross-validation,
the same KFold seeds, and the same median-imputation pipeline. The Gradient
Boosting column of S4 therefore reproduces Main Table 2 digit for digit, and that
agreement serves as the table's own reproducibility check.

Requires the full processed dataset, which is not distributed with this repository.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from utils.io import load_config, read_parquet          # noqa: E402
from utils.validation import random_cv                  # noqa: E402
import s11_controlled_analysis as s11                    # noqa: E402

from sklearn.pipeline import Pipeline                    # noqa: E402
from sklearn.impute import SimpleImputer                 # noqa: E402

VARIANT = "traj_wind"
S4_CELLS = [(16, "M", "Male"), (18, "M", "Male"), (18, "F", "Female")]
ALL_CELLS = [(c, s, lab) for c in (16, 17, 18) for s, lab in (("M", "Male"), ("F", "Female"))]

config = load_config()
interim = Path(config["paths"]["interim"])
processed = Path(config["paths"]["processed"])
records = read_parquet(interim / "cleaned_records.parquet")
athletes = read_parquet(interim / "cleaned_athletes.parquet")
gl = read_parquet(processed / "group_labels.parquet")
rec_100m = records[records["event"] == "100m"].copy()

n_folds = config["modeling"]["cv"]["n_folds"]
seed = config["modeling"]["cv"]["random_seed"]
n_rep = config["modeling"]["cv"]["n_repeats"]
n_jobs = config["modeling"]["cv"].get("n_jobs", 1)
registry = s11.load_feature_registry()

print(f"records {len(records):,} | 100 m {len(rec_100m):,} | group_labels {len(gl):,}")
print(f"CV: {n_folds} folds x {n_rep} repeats, seed={seed}, n_jobs={n_jobs}\n")

# ======================================================================
# S13: province groups
# ======================================================================
print("=" * 78)
print("Supplementary Table S13")
print("=" * 78)
prov = athletes.drop_duplicates("athlete_id").set_index("athlete_id")["province"]
rows13 = []
for cutoff, sx, lab in ALL_CELLS:
    ids = gl[(gl[f"cutoff_{cutoff}_group"] == "A") & (gl["sex"] == sx)]["athlete_id"]
    p = prov.reindex(ids).dropna()
    counts = p.value_counts()
    kept = counts[counts >= 5]
    rows13.append(dict(cell=f"{lab} {cutoff}", n_group_a=len(ids),
                       provinces_represented=int((counts > 0).sum()),
                       province_groups_used=int(len(kept)),
                       n_in_validation=int(kept.sum())))
s13 = pd.DataFrame(rows13)
print(s13.to_string(index=False))
print("\n(Published values: 30/30/31 provinces for males and 30/31/30 for females; "
      "24/25/28 and 23/25/25 groups; 794/2366/2175 and 396/584/470 athletes in validation)")
s13.to_csv("results/tables/s13_province_groups.csv", index=False)
print("-> results/tables/s13_province_groups.csv\n")

# ======================================================================
# S4: three boosting frameworks
# ======================================================================
print("=" * 78)
print("Supplementary Table S4")
print("=" * 78)
missing = []
try:
    from xgboost import XGBRegressor
except ImportError:
    missing.append("xgboost")
try:
    from lightgbm import LGBMRegressor
except ImportError:
    missing.append("lightgbm")
if missing:
    print("Missing dependencies: " + ", ".join(missing))
    print("Install them and re-run for S4:  pip install " + " ".join(missing))
    print("(S13 above is already complete and can be used as printed.)")
    sys.exit(0)

gb_params = config["modeling"]["algorithms"]["gradient_boosting"]
n_est = gb_params["n_estimators"]
depth = gb_params["max_depth"]
lr = gb_params["learning_rate"]
print(f"Shared settings for all three frameworks: n_estimators={n_est}, "
      f"max_depth={depth}, learning_rate={lr}\n")


def imp(m):
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", m)])


models = {
    "GradientBoosting": s11.get_models(config)["GradientBoosting"],
    "XGBoost": imp(XGBRegressor(n_estimators=n_est, max_depth=depth, learning_rate=lr,
                                random_state=seed, n_jobs=1, verbosity=0)),
    "LightGBM": imp(LGBMRegressor(n_estimators=n_est, max_depth=depth, learning_rate=lr,
                                  num_leaves=2 ** depth - 1, random_state=seed,
                                  n_jobs=1, verbose=-1)),
}

rows4 = []
for cutoff, sx, lab in S4_CELLS:
    ids = gl[(gl[f"cutoff_{cutoff}_group"] == "A") & (gl["sex"] == sx)]["athlete_id"].tolist()
    feats = s11.build_features_for_group_a(
        ids, rec_100m, athletes, cutoff, config["features"]["min_records_for_slope"])
    target = rec_100m.groupby("athlete_id")["time_raw"].min()
    feats["target"] = target.reindex(feats.index)
    feats = feats.dropna(subset=["target"])
    cols = [c for c in s11.get_variant_cols(registry, VARIANT) if c in feats.columns]
    X = feats[cols].values.astype(np.float64)
    y = feats["target"].values.astype(np.float64)
    row = dict(cutoff=cutoff, sex=lab, n=len(y))
    for name, mdl in models.items():
        r = random_cv(mdl, X, y, n_folds, n_rep, seed, n_jobs=n_jobs)
        row[name] = f"{r['R2_mean']:.3f} [{r['R2_ci_lo']:.3f}, {r['R2_ci_hi']:.3f}]"
        row[name + "_raw"] = r["R2_mean"]
        print(f"  {lab} {cutoff}  {name:<17} n={len(y):<5} {row[name]}", flush=True)
    rows4.append(row)

s4 = pd.DataFrame(rows4)
s4.to_csv("results/tables/s4_algorithm_comparison.csv", index=False)
print("\n-> results/tables/s4_algorithm_comparison.csv")

# Self-check: the GB column must match Main Table 2 digit for digit.
print("\nSelf-check: GB column against Main Table 2")
t2 = pd.read_csv("results/tables/table2_group_a_performance.csv")
t2 = t2[(t2.variant == VARIANT) & (t2.model == "GradientBoosting") & (t2.cv_strategy == "random")]
ok = True
for r in rows4:
    pub = t2[(t2.cutoff_age == r["cutoff"]) & (t2.sex == r["sex"])]["R2_mean"].iloc[0]
    d = abs(pub - r["GradientBoosting_raw"])
    flag = "" if d < 1e-9 else "   <-- mismatch"
    if d >= 1e-9:
        ok = False
    print(f"  {r['sex']} {r['cutoff']}: here {r['GradientBoosting_raw']:.6f} | "
          f"Main Table 2 {pub:.6f} | diff {d:.2e}{flag}")
print("\n" + ("GB column matches Main Table 2 exactly; S4 is usable" if ok
              else "GB column does not match Main Table 2; do not use these values"))
