"""
Reviewer 4, major comment #3 -- outcome recomputed with the mechanical
predictor-outcome identity removed.
======================================================================
Group A is defined in the manuscript (paragraph 33) as athletes with at least
one 100 m record AFTER the cutoff, that is by career continuation. It is not
defined by whether the lifetime personal best has already been set. For a
substantial and cutoff-dependent share of Group A the lifetime PB therefore
IS one of the pre-cutoff races, so the target equals the feature
`best_time_raw` exactly and the model is evaluating an identity rather than
making a prediction:

    cutoff   16      17      18
    male    22.7%   31.9%   43.4%
    female  32.3%   36.7%   46.2%

That share rises with the cutoff age in the same direction as the reported R2
(0.606 -> 0.737 -> 0.799 for males), so it is a candidate alternative
explanation for the developmental trend in paragraph 56. Paragraph 71 argues
that the dependence is "present to the same degree" across target populations;
that holds for the cross-population contrast at a fixed cutoff, but plainly not
across cutoffs.

This script splits Group A at each cutoff into

    target_is_feature      lifetime_pb == min(pre-cutoff time), to within 1e-9
    genuine_post_cutoff    lifetime_pb set strictly after the cutoff

and fits the published specification (GradientBoosting, traj_wind, random
5-fold CV, identical seeds) to each, plus to the pooled sample as a check.

The `all` rows must reproduce Main Table 2. The script verifies this and refuses
to vouch for the subgroup numbers otherwise.

Usage:  python3 r4_reanalysis/r4_03_overlap_split.py [--repeats N] [--cells M16,...]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent   # repo root; this script lives in r4_reanalysis/
sys.path.insert(0, str(HERE / "src"))

CELLS = [(16, "M", "Male"), (16, "F", "Female"),
         (17, "M", "Male"), (17, "F", "Female"),
         (18, "M", "Male"), (18, "F", "Female")]
VARIANT = "traj_wind"
MODEL = "GradientBoosting"


def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


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
        print("r4_03_overlap_split.py -- Reviewer 4 comment #3: outcome recomputed "
              "with the\nmechanical predictor-outcome identity removed.")
        print("\nThis script requires the full processed dataset, which is not "
              "distributed with\nthis repository because it contains athlete names "
              "and dates of birth.\n\nMissing:")
        for p in missing:
            print(f"  {p.relative_to(HERE)}")
        print("\nThe full dataset is available from the corresponding author on "
              "reasonable request\nunder a data-use agreement. To run the modelling "
              "code on the public de-identified\nsample instead, use "
              "scripts/demo_reproduce.py.")
        raise SystemExit(0)


def main() -> None:
    require_full_dataset()

    from utils.io import load_config, read_parquet
    from utils.validation import random_cv
    import s11_controlled_analysis as s11

    n_rep = int(arg("--repeats", 500))
    only = arg("--cells", None)
    cells = CELLS if not only else [
        c for c in CELLS if f"{c[1]}{c[0]}" in set(only.split(","))]

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
    model = s11.get_models(config)[MODEL]

    rows = []
    for cutoff, sex_code, sex_label in cells:
        best_pre = (rec_100m[rec_100m["age_at_comp"] <= cutoff]
                    .groupby("athlete_id")["time_raw"].min())

        a_ids = group_labels[
            (group_labels[f"cutoff_{cutoff}_group"] == "A")
            & (group_labels["sex"] == sex_code)
        ]["athlete_id"].tolist()

        feats = s11.build_features_for_group_a(
            a_ids, rec_100m, athletes, cutoff,
            config["features"]["min_records_for_slope"],
        )
        feats["lifetime_pb"] = group_labels.set_index("athlete_id")["lifetime_pb"].loc[feats.index]
        feats = feats.dropna(subset=["lifetime_pb"])

        identity = (feats["lifetime_pb"] - best_pre.reindex(feats.index)).abs() < 1e-9
        cols = [c for c in s11.get_variant_cols(registry, VARIANT) if c in feats.columns]

        for label, mask in [
            ("all", pd.Series(True, index=feats.index)),
            ("target_is_feature", identity),
            ("genuine_post_cutoff", ~identity),
        ]:
            sub = feats[mask]
            if len(sub) < 40:
                print(f"  {sex_label[:1]}{cutoff} {label}: n={len(sub)} < 40, skipped")
                continue
            t0 = time.time()
            r = random_cv(model, sub[cols].values.astype(np.float64),
                          sub["lifetime_pb"].values.astype(np.float64),
                          n_folds, n_rep, seed, n_jobs=n_jobs)
            rows.append({
                "cutoff_age": cutoff, "sex": sex_label, "subgroup": label,
                "n": len(sub), "pct_of_group_a": round(100 * len(sub) / len(feats), 1),
                "R2_mean": r["R2_mean"], "R2_ci_lo": r["R2_ci_lo"],
                "R2_ci_hi": r["R2_ci_hi"], "RMSE_mean": r["RMSE_mean"],
                "n_repeats": n_rep,
            })
            print(f"  {sex_label[:1]}{cutoff} {label:<20} n={len(sub):<5} "
                  f"R2={r['R2_mean']:.4f} [{r['R2_ci_lo']:.3f}, {r['R2_ci_hi']:.3f}]  "
                  f"{time.time()-t0:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    dest = HERE / "results/tables/r4_overlap_split.csv"
    out.to_csv(dest, index=False)
    print(f"\nSaved {dest}")

    # ---- self-check: the pooled rows must equal the published table ----------
    pub = pd.read_csv(HERE / "results/tables/table2_group_a_performance.csv")
    pub = pub[(pub.variant == VARIANT) & (pub.model == MODEL)
              & (pub.cv_strategy == "random")]
    print(f"\n{'cell':<8}{'pooled here':>14}{'published':>13}{'diff':>12}")
    print("-" * 47)
    bad = 0
    for _, r in out[out.subgroup == "all"].iterrows():
        p = pub[(pub.cutoff_age == r.cutoff_age) & (pub.sex == r.sex)]
        if not len(p):
            continue
        d = r.R2_mean - p.R2_mean.iloc[0]
        if abs(d) > 1e-6:
            bad += 1
        print(f"{r.sex[:1]}{r.cutoff_age:<7}{r.R2_mean:>14.6f}{p.R2_mean.iloc[0]:>13.6f}"
              f"{d:>12.2e}{'   <-- MISMATCH' if abs(d) > 1e-6 else ''}")
    print("\n" + ("BASELINE CHECK PASSED" if not bad else
                  f"BASELINE CHECK FAILED ({bad}) - do not use the subgroup numbers"))
    if n_rep != 500:
        print("NOTE: --repeats != 500, the pooled rows will not match the published "
              "table exactly.")

    piv = out.pivot_table(index=["sex", "cutoff_age"], columns="subgroup",
                          values="R2_mean").round(3)
    print("\n" + piv.to_string())
    print("\nRead the `genuine_post_cutoff` column against `all`: if the "
          "developmental trend\nsurvives there, the rising share of mechanical "
          "identities does not produce it.")


if __name__ == "__main__":
    main()
