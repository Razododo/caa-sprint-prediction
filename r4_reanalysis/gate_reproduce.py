"""
REPRODUCTION GATE
=================
Re-runs one published cell of s11_controlled_analysis.py and checks it against
results/tables/table2_group_a_performance.csv.

If this does not reproduce to ~6 decimals, NOTHING computed in this container is
commensurable with the published tables, and every new sensitivity analysis would
introduce a fresh inconsistency of exactly the kind Reviewer 4 flagged.

Usage:  python3 gate_reproduce.py [--full]
        (default checks Ridge only, which is fast; --full adds GradientBoosting)
"""
import sys
import time
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent   # repo root; this script lives in r4_reanalysis/
sys.path.insert(0, str(HERE / "src"))

from utils.io import load_config, read_parquet                      # noqa: E402
from utils.validation import random_cv                              # noqa: E402
import s11_controlled_analysis as s11                               # noqa: E402


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


def build_cell(cutoff: int, sex_code: str, variant: str):
    """Rebuild exactly what s11.main() builds for one (cutoff, sex, variant) cell."""
    config = load_config()
    registry = s11.load_feature_registry()

    interim = Path(config["paths"]["interim"])
    processed = Path(config["paths"]["processed"])

    records = read_parquet(interim / "cleaned_records.parquet")
    athletes = read_parquet(interim / "cleaned_athletes.parquet")
    group_labels = read_parquet(processed / "group_labels.parquet")

    rec_100m = records[records["event"] == "100m"].copy()

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
    feat_cols = s11.get_variant_cols(registry, variant)
    avail = [c for c in feat_cols if c in feats.columns]
    X = feats[avail].values.astype(np.float64)

    return config, X, y, avail


def main() -> int:
    require_full_dataset()
    full = "--full" in sys.argv
    cutoff, sex_code, sex_label, variant = 16, "M", "Male", "traj_wind"

    published = pd.read_csv(HERE / "results/tables/table2_group_a_performance.csv")
    config, X, y, avail = build_cell(cutoff, sex_code, variant)

    n_folds = config["modeling"]["cv"]["n_folds"]
    seed = config["modeling"]["cv"]["random_seed"]
    n_jobs = config["modeling"]["cv"].get("n_jobs", 1)
    n_rep = 500

    print(f"\nCell: cutoff {cutoff} {sex_label} / {variant}")
    print(f"  rebuilt  n_samples={X.shape[0]}  n_features={X.shape[1]}")

    ref_row = published[
        (published.cutoff_age == cutoff) & (published.sex == sex_label)
        & (published.variant == variant)
    ]
    print(f"  published n_samples={int(ref_row.n_samples.iloc[0])}"
          f"  n_features={int(ref_row.n_features.iloc[0])}")

    models = s11.get_models(config)
    to_check = ["Ridge"] + (["GradientBoosting"] if full else [])

    ok = True
    print(f"\n{'model':<18}{'metric':<10}{'reproduced':>13}{'published':>13}{'diff':>12}")
    print("-" * 66)
    for model_name in to_check:
        t0 = time.time()
        r = random_cv(models[model_name], X, y, n_folds, n_rep, seed, n_jobs=n_jobs)
        el = time.time() - t0
        ref = ref_row[ref_row.model == model_name].iloc[0]
        for metric, got, exp in [
            ("R2_mean", r["R2_mean"], ref.R2_mean),
            ("R2_ci_lo", r["R2_ci_lo"], ref.R2_ci_lo),
            ("R2_ci_hi", r["R2_ci_hi"], ref.R2_ci_hi),
            ("RMSE_mean", r["RMSE_mean"], ref.RMSE_mean),
        ]:
            d = got - exp
            flag = "" if abs(d) < 1e-6 else "   <-- MISMATCH"
            if abs(d) >= 1e-6:
                ok = False
            print(f"{model_name:<18}{metric:<10}{got:>13.6f}{exp:>13.6f}{d:>12.2e}{flag}")
        print(f"{'':<18}({el:.1f}s)")

    print("\n" + ("GATE PASSED - container reproduces published values exactly"
                  if ok else "GATE FAILED - do NOT compute new analyses here"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
