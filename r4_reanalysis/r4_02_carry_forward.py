"""
Reviewer 4, major comment #2 -- carry-forward baseline.
=======================================================
R4 asks whether the model does anything beyond carrying forward the athlete's
best pre-cutoff time. Three predictors are compared on IDENTICAL CV folds
(same KFold seeds 42..541, 5 folds) so every number here is commensurable with
Main Table 2:

  naive     y_hat = best_time_raw                (no fitting at all)
  ols1      y_hat = OLS(lifetime_pb ~ best_time_raw), cross-validated
  full      the published traj_wind model         (Ridge / GB)

Output: results/tables/r4_carry_forward_baseline.csv
"""
import sys
import time
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent.parent   # repo root; this script lives in r4_reanalysis/
sys.path.insert(0, str(HERE / "src"))

from utils.io import load_config, read_parquet          # noqa: E402
from utils.validation import random_cv                  # noqa: E402
import s11_controlled_analysis as s11                    # noqa: E402

CUTOFFS = [16, 17, 18]
SEXES = [("M", "Male"), ("F", "Female")]
def require_full_dataset() -> None:
    """
    The analyses added at peer review need the full processed dataset, which
    contains personal data and is not distributed with this repository (see the
    Data section of the README). Exit with an explanation rather than a
    traceback, matching the behaviour of redraw_figures_4_5.py and
    s14_composition_effect.py.
    """
    needed = [
        HERE / "data/interim/cleaned_records.parquet",
        HERE / "data/interim/cleaned_athletes.parquet",
        HERE / "data/processed/group_labels.parquet",
    ]
    missing = [p for p in needed if not p.exists()]
    if missing:
        print("%s -- analysis added at peer review." % pathlib.Path(__file__).name)
        print("\nThis script requires the full processed dataset, which is not "
              "distributed with\nthis repository because it contains athlete names "
              "and dates of birth.\n\nMissing:")
        for p in missing:
            print("  %s" % p.relative_to(HERE))
        print("\nThe full dataset is available from the corresponding author on "
              "reasonable request\nunder a data-use agreement. To run the modelling "
              "code on the public de-identified\nsample instead, use "
              "scripts/demo_reproduce.py.")
        raise SystemExit(0)


VARIANT = "traj_wind"


def naive_cv(x: np.ndarray, y: np.ndarray, n_folds: int, n_repeats: int, seed: int):
    """R2 of the un-fitted predictor y_hat = x, evaluated on the same folds."""
    vals, rmses = [], []
    for s in range(seed, seed + n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=s)
        for _, test_idx in kf.split(x):
            vals.append(r2_score(y[test_idx], x[test_idx]))
            rmses.append(np.sqrt(mean_squared_error(y[test_idx], x[test_idx])))
    v = np.array(vals)
    return {
        "R2_mean": v.mean(),
        "R2_ci_lo": np.percentile(v, 2.5),
        "R2_ci_hi": np.percentile(v, 97.5),
        "RMSE_mean": float(np.mean(rmses)),
    }


def main() -> None:
    require_full_dataset()
    quick = "--quick" in sys.argv
    n_rep = 20 if quick else 500

    config = load_config()
    registry = s11.load_feature_registry()
    interim = Path(config["paths"]["interim"])
    processed = Path(config["paths"]["processed"])

    records = read_parquet(interim / "cleaned_records.parquet")
    athletes = read_parquet(interim / "cleaned_athletes.parquet")
    group_labels = read_parquet(processed / "group_labels.parquet")
    rec_100m = records[records["event"] == "100m"].copy()

    n_folds = config["modeling"]["cv"]["n_folds"]
    seed = config["modeling"]["cv"]["random_seed"]
    n_jobs = config["modeling"]["cv"].get("n_jobs", 1)
    models = s11.get_models(config)

    ols1 = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LinearRegression()),
    ])

    rows = []
    for cutoff in CUTOFFS:
        for sex_code, sex_label in SEXES:
            col = f"cutoff_{cutoff}_group"
            a_ids = group_labels[
                (group_labels[col] == "A") & (group_labels["sex"] == sex_code)
            ]["athlete_id"].tolist()

            feats = s11.build_features_for_group_a(
                a_ids, rec_100m, athletes, cutoff,
                config["features"]["min_records_for_slope"],
            )
            pb_map = group_labels.set_index("athlete_id")["lifetime_pb"]
            feats["lifetime_pb"] = pb_map.loc[feats.index]
            feats = feats.dropna(subset=["lifetime_pb"])

            y = feats["lifetime_pb"].values.astype(np.float64)
            cols = [c for c in s11.get_variant_cols(registry, VARIANT) if c in feats.columns]
            X = feats[cols].values.astype(np.float64)
            x1 = feats[["best_time_raw"]].values.astype(np.float64)

            # 1. naive carry-forward (no fitting)
            xflat = np.where(np.isnan(x1[:, 0]), np.nanmedian(x1), x1[:, 0])
            r_naive = naive_cv(xflat, y, n_folds, n_rep, seed)
            rows.append(dict(cutoff_age=cutoff, sex=sex_label, n=len(y),
                             predictor="naive_carry_forward", model="none", **r_naive))

            # 2. recalibrated single-predictor OLS
            t0 = time.time()
            r_ols = random_cv(ols1, x1, y, n_folds, n_rep, seed, n_jobs=n_jobs)
            rows.append(dict(cutoff_age=cutoff, sex=sex_label, n=len(y),
                             predictor="ols_best_time_only", model="OLS",
                             R2_mean=r_ols["R2_mean"], R2_ci_lo=r_ols["R2_ci_lo"],
                             R2_ci_hi=r_ols["R2_ci_hi"], RMSE_mean=r_ols["RMSE_mean"]))

            # 3. published full models
            for mname in ["Ridge", "GradientBoosting"]:
                r = random_cv(models[mname], X, y, n_folds, n_rep, seed, n_jobs=n_jobs)
                rows.append(dict(cutoff_age=cutoff, sex=sex_label, n=len(y),
                                 predictor=f"full_{VARIANT}", model=mname,
                                 R2_mean=r["R2_mean"], R2_ci_lo=r["R2_ci_lo"],
                                 R2_ci_hi=r["R2_ci_hi"], RMSE_mean=r["RMSE_mean"]))

            print(f"cutoff {cutoff} {sex_label} (n={len(y)}) done in {time.time()-t0:.0f}s",
                  flush=True)

    out = pd.DataFrame(rows)
    dest = HERE / "results/tables/r4_carry_forward_baseline.csv"
    out.to_csv(dest, index=False)
    print(f"\nSaved {dest}")

    piv = out.pivot_table(index=["sex", "cutoff_age"],
                          columns="predictor", values="R2_mean").round(3)
    print("\n" + piv.to_string())


if __name__ == "__main__":
    main()
