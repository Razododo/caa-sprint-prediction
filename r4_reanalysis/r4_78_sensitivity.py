"""
Reviewer 4 -- sensitivity analyses #7 (wind eligibility) and #8 (100 m hurdles).
================================================================================
Every scenario is run alongside the published baseline in the SAME execution, on
the SAME folds, so the `baseline` rows must reproduce Main Table 2 exactly. If
they do not, the run is invalid and the variant numbers must not be used.

Group A membership is held at its published assignment in all scenarios, so each
contrast isolates the effect of the measurement change rather than confounding it
with a change in who is in the sample. The number of athletes whose Group A
status *would* change is reported separately.

Scenarios
---------
hurdles   Drop the 2,444 records inside event == '100m' whose event_full names a
          hurdles event, then recompute lifetime PB and all trajectory features.
          These times (11.08-19.98 s) all pass the 8.0-20.0 s validity filter in
          paragraph 31, so no existing exclusion removes them.

wind      Three targets:
            pb_raw            min(time_raw)                       [published]
            pb_wc             min(time_wind_corrected)            [Linthorne]
            pb_legal_only     min(time_raw) over races with wind <= +2.0 m/s
          This addresses both halves of R4's question: whether PB should have a
          wind-eligibility rule, and why the outcome is a raw time while the
          predictors are wind-corrected.

Usage:  python3 r4_78_sensitivity.py --mode hurdles [--repeats N] [--cells M16,...]
        python3 r4_78_sensitivity.py --mode wind    [--repeats N] [--cells M16,...]
"""
import sys
import time
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent   # repo root; this script lives in r4_reanalysis/
sys.path.insert(0, str(HERE / "src"))

from utils.io import load_config, read_parquet      # noqa: E402
from utils.validation import random_cv              # noqa: E402
import s11_controlled_analysis as s11                # noqa: E402

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
MODELS = ["Ridge", "GradientBoosting"]
CELLS = [(16, "M", "Male"), (16, "F", "Female"),
         (17, "M", "Male"), (17, "F", "Female"),
         (18, "M", "Male"), (18, "F", "Female")]
WIND_LEGAL = 2.0


def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def is_hurdles(rec: pd.DataFrame) -> pd.Series:
    return rec["event_full"].astype(str).str.contains("栏", na=False)


def build_scenarios(rec_100m: pd.DataFrame, mode: str):
    """Return {scenario_name: (records_for_features, target_series)}."""
    pb_raw = rec_100m.groupby("athlete_id")["time_raw"].min()

    if mode == "hurdles":
        h = is_hurdles(rec_100m)
        clean = rec_100m[~h].copy()
        print(f"  hurdles rows dropped: {int(h.sum()):,} "
              f"({rec_100m.loc[h, 'athlete_id'].nunique():,} athletes)")
        pb_clean = clean.groupby("athlete_id")["time_raw"].min()
        changed = (pb_raw.reindex(pb_clean.index) - pb_clean).abs() > 1e-9
        print(f"  athletes whose lifetime PB changes: {int(changed.sum()):,}")
        return {
            "baseline": (rec_100m, pb_raw),
            "no_hurdles": (clean, pb_clean),
        }

    if mode == "wind":
        wc = rec_100m.dropna(subset=["time_wind_corrected"])
        pb_wc = wc.groupby("athlete_id")["time_wind_corrected"].min()
        legal = rec_100m[rec_100m["wind_speed"].notna()
                         & (rec_100m["wind_speed"] <= WIND_LEGAL)]
        pb_legal = legal.groupby("athlete_id")["time_raw"].min()
        print(f"  records with known wind: {len(wc):,} / {len(rec_100m):,}")
        print(f"  athletes losing a PB under wind<=+2.0 rule: "
              f"{len(pb_raw) - len(pb_raw.index.intersection(pb_legal.index)):,}")
        return {
            "baseline": (rec_100m, pb_raw),
            "pb_wind_corrected": (rec_100m, pb_wc),
            "pb_legal_wind_only": (rec_100m, pb_legal),
        }

    raise SystemExit(f"unknown mode {mode!r}")


def main() -> None:
    require_full_dataset()
    mode = arg("--mode", None)
    if mode is None:
        raise SystemExit(__doc__)
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
    models = s11.get_models(config)

    print(f"scenario set: {mode}")
    scenarios = build_scenarios(rec_100m, mode)

    rows = []
    for cutoff, sex_code, sex_label in cells:
        a_ids = group_labels[
            (group_labels[f"cutoff_{cutoff}_group"] == "A")
            & (group_labels["sex"] == sex_code)
        ]["athlete_id"].tolist()

        # Build every scenario's design matrix first, so the athletes common to
        # all of them can be identified. Dropping records can remove an athlete
        # from the sample entirely (no pre-cutoff flat-100 m record left), which
        # would otherwise confound a measurement effect with a change in sample
        # composition -- the very thing this paper is about.
        built = {}
        for sname, (srec, target) in scenarios.items():
            feats = s11.build_features_for_group_a(
                a_ids, srec, athletes, cutoff,
                config["features"]["min_records_for_slope"],
            )
            feats["target"] = target.reindex(feats.index)
            feats = feats.dropna(subset=["target"])
            built[sname] = feats

        common = None
        for f in built.values():
            common = f.index if common is None else common.intersection(f.index)
        n_full = len(built["baseline"])
        print(f"  {sex_label[:1]}{cutoff}: baseline n={n_full}, "
              f"common to all scenarios n={len(common)} "
              f"({n_full - len(common)} athletes lost by the variant)")

        for sname, feats in built.items():
            for sample_def, idx in [("full", feats.index), ("matched", common)]:
                t0 = time.time()
                sub = feats.loc[idx]
                y = sub["target"].values.astype(np.float64)
                cols = [c for c in s11.get_variant_cols(registry, VARIANT)
                        if c in sub.columns]
                X = sub[cols].values.astype(np.float64)

                for mname in MODELS:
                    r = random_cv(models[mname], X, y, n_folds, n_rep, seed,
                                  n_jobs=n_jobs)
                    rows.append({
                        "mode": mode, "scenario": sname, "sample": sample_def,
                        "cutoff_age": cutoff, "sex": sex_label, "model": mname,
                        "n": len(y),
                        "R2_mean": r["R2_mean"], "R2_ci_lo": r["R2_ci_lo"],
                        "R2_ci_hi": r["R2_ci_hi"], "RMSE_mean": r["RMSE_mean"],
                        "n_repeats": n_rep,
                    })
                print(f"    {sname:<20} {sample_def:<8} n={len(y):<6} "
                      f"{time.time()-t0:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    dest = HERE / f"results/tables/r4_sensitivity_{mode}.csv"
    out.to_csv(dest, index=False)
    print(f"\nSaved {dest}")

    # ---- self-check: baseline must equal the published table -----------------
    pub = pd.read_csv(HERE / "results/tables/table2_group_a_performance.csv")
    pub = pub[(pub.variant == VARIANT) & (pub.cv_strategy == "random")]
    print(f"\n{'cell':<8}{'model':<18}{'baseline here':>15}{'published':>13}{'diff':>12}")
    print("-" * 66)
    bad = 0
    for _, r in out[(out.scenario == "baseline") & (out["sample"] == "full")].iterrows():
        p = pub[(pub.cutoff_age == r.cutoff_age) & (pub.sex == r.sex)
                & (pub.model == r.model)]
        if not len(p):
            continue
        d = r.R2_mean - p.R2_mean.iloc[0]
        if abs(d) > 1e-6:
            bad += 1
        print(f"{r.sex[:1]}{r.cutoff_age:<7}{r.model:<18}{r.R2_mean:>15.6f}"
              f"{p.R2_mean.iloc[0]:>13.6f}{d:>12.2e}"
              f"{'   <-- MISMATCH' if abs(d) > 1e-6 else ''}")
    print("\n" + ("BASELINE CHECK PASSED" if not bad else
                  f"BASELINE CHECK FAILED ({bad}) - do not use these variant numbers"))
    if n_rep != 500:
        print("NOTE: --repeats != 500, baseline will not match the published table exactly.")

    piv = out.pivot_table(index=["sex", "cutoff_age", "model", "sample"],
                          columns="scenario", values="R2_mean").round(4)
    print("\n" + piv.to_string())


if __name__ == "__main__":
    main()
