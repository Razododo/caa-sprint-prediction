"""
End-to-end demo on the public de-identified sample.

Runs the same modelling code used for the paper (src/utils/validation.py,
src/s06_model_train.py model definitions) on data/sample/, so that a reader
without access to the full dataset can verify the pipeline executes and see
the shape of the results.

By default it restricts to Group A (athletes still improving after the cutoff),
which is the paper's primary analysis set. Passing --group all reproduces the
inflated all-athlete estimate discussed in the manuscript, where Group B
athletes have already achieved their lifetime PB before the cutoff and are
therefore near-trivially predictable.

The sample is small and deliberately over-samples Group A, so the R^2 values
printed here are noisier than the published estimates; this script demonstrates
the procedure, it does not reproduce the paper's point estimates.

Usage:
    python scripts/demo_reproduce.py
    python scripts/demo_reproduce.py --cutoff 22 --group all --n-repeats 20
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml  # noqa: E402

from s06_model_train import get_feature_columns, get_models  # noqa: E402
from utils.validation import random_cv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int, default=18, choices=[16, 18, 20, 22])
    parser.add_argument("--variant", default="traj_wind")
    parser.add_argument("--group", default="A", choices=["A", "all"])
    parser.add_argument("--n-repeats", type=int, default=20)
    args = parser.parse_args()

    config = yaml.safe_load((PROJECT_ROOT / "config" / "config.yaml").read_text())
    registry = yaml.safe_load(
        (PROJECT_ROOT / "config" / "feature_registry.yaml").read_text()
    )

    sample_fp = PROJECT_ROOT / "data" / "sample" / f"features_cutoff_{args.cutoff}_sample.csv"
    if not sample_fp.exists():
        sys.exit(f"Missing {sample_fp}. See README for how the sample is built.")

    df = pd.read_csv(sample_fp)

    if args.group == "A":
        labels = pd.read_csv(PROJECT_ROOT / "data" / "sample" / "group_labels_sample.csv")
        col = f"cutoff_{args.cutoff}_group"
        keep = set(labels.loc[labels[col] == "A", "athlete_id"])
        df = df[df["athlete_id"].isin(keep)]

    seed = config["modeling"]["cv"]["random_seed"]
    n_folds = config["modeling"]["cv"]["n_folds"]
    feat_cols = get_feature_columns(registry, args.variant)

    print(f"Sample: {sample_fp.name}  ({len(df)} athletes, one row per athlete)")
    print(f"Group: {args.group}   variant: {args.variant}")
    print(f"seed={seed}   folds={n_folds}   repeats={args.n_repeats}")
    print("Cross-validation is athlete-disjoint by construction (one row per athlete).\n")

    for sex_label, sex_code in [("Male", "M"), ("Female", "F")]:
        sub = df[df["sex"] == sex_code]
        available = [c for c in feat_cols if c in sub.columns]
        X = sub[available].to_numpy(dtype=np.float64)
        y = sub["lifetime_pb"].to_numpy(dtype=np.float64)
        valid = np.isfinite(y)
        X, y = X[valid], y[valid]

        if len(y) < 50:
            print(f"{sex_label}: n={len(y)} — too small for CV, skipped")
            continue

        for name, model in get_models(config).items():
            r = random_cv(model, X, y, n_folds, args.n_repeats, seed, n_jobs=1)
            print(
                f"{sex_label:6s} n={len(y):4d}  {name:17s} "
                f"R2={r['R2_mean']:.3f} [{r['R2_ci_lo']:.3f}, {r['R2_ci_hi']:.3f}]  "
                f"RMSE={r['RMSE_mean']:.3f}"
            )
        print()


if __name__ == "__main__":
    main()
