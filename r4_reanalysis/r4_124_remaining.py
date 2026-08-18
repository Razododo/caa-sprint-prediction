"""
Reviewer 4 -- the three requests not covered by the earlier scripts.
====================================================================
A.  Major comment 1, third bullet: "Reanalyse future performance using a
    clearly post-cutoff outcome, preferably post-cutoff best or improvement
    within a standardized follow-up period."

    r4_03_overlap_split.py removed the athletes whose lifetime best is already
    fixed before the cutoff. That is a related analysis but not the one asked
    for, because it drops those athletes instead of giving them a genuinely
    prospective target. Here the outcome itself is replaced, for the whole of
    Group A:

        pb_lifetime     min(time_raw) over all records          [published]
        pb_post_cutoff  min(time_raw) over records after the cutoff
        improvement_2y  best pre-cutoff minus best within 2 years after it
                        (positive = faster; athletes with no race in the
                        window are dropped, and the count is reported)

B.  Major comment 2: "Please provide a balanced-panel analysis restricted to
    athletes eligible and observable at all main cutoffs."

    Group A membership at all of 16, 17 and 18 leaves 95 males and 84 females.
    The analysis is run and reported in full. Intervals at that sample size are
    very wide, which is itself the answer: the balanced panel cannot adjudicate
    the cutoff-to-cutoff comparison. Compare Supplementary Table S3 Panel C,
    where n = 95 gives R2 = 0.091 with a 95% interval of [-0.957, 0.529].

C.  Major comment 6: confidence intervals for "anthropometric delta-R2,
    feature-ablation differences, and random versus province-disjoint
    validation", by paired resampling on identical folds.

    r4_06 covered the feature-ablation and algorithm contrasts. The
    anthropometric contrast (Table 3, v2 versus v3 on the anthropometric
    subsample) is done here: identical athletes, identical folds, so the paired
    interval is far narrower than the two marginal intervals in Table 3, which
    span roughly [0.16, 0.71].

    Random versus province-disjoint validation is NOT paired here, and cannot
    be: KFold and GroupKFold generate different partitions by construction, so
    there are no shared folds to pair on. That limitation is stated rather than
    worked around.

Usage:  python3 r4_reanalysis/r4_124_remaining.py [--repeats N] [--only A,B,C]
"""
import pathlib
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent.parent   # repo root; this script lives in r4_reanalysis/
sys.path.insert(0, str(HERE / "src"))

CUTOFFS = [16, 17, 18]
SEXES = [("M", "Male"), ("F", "Female")]
VARIANT = "traj_wind"
MODELS = ["Ridge", "GradientBoosting"]
FOLLOWUP_YEARS = 2.0


def require_full_dataset() -> None:
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


def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main() -> None:
    require_full_dataset()

    from utils.io import load_config, read_parquet
    from utils.validation import random_cv
    import s11_controlled_analysis as s11

    n_rep = int(arg("--repeats", 500))
    only = set(arg("--only", "A,B,C").split(","))

    config = load_config()
    registry = s11.load_feature_registry()
    interim = Path(config["paths"]["interim"])
    processed = Path(config["paths"]["processed"])

    records = read_parquet(interim / "cleaned_records.parquet")
    athletes = read_parquet(interim / "cleaned_athletes.parquet")
    gl = read_parquet(processed / "group_labels.parquet")
    rec = records[records["event"] == "100m"].copy()

    n_folds = config["modeling"]["cv"]["n_folds"]
    seed = config["modeling"]["cv"]["random_seed"]
    n_jobs = config["modeling"]["cv"].get("n_jobs", 1)
    models = s11.get_models(config)
    min_slope = config["features"]["min_records_for_slope"]
    tdir = HERE / "results/tables"

    def build(ids, cutoff):
        f = s11.build_features_for_group_a(ids, rec, athletes, cutoff, min_slope)
        return f

    def cols_for(f, variant):
        return [c for c in s11.get_variant_cols(registry, variant) if c in f.columns]

    # =====================================================================
    # A.  post-cutoff outcomes
    # =====================================================================
    if "A" in only:
        print("\n" + "=" * 74)
        print("A.  Reviewer 4, comment 1: genuinely post-cutoff outcomes")
        print("=" * 74)
        rows = []
        for cutoff in CUTOFFS:
            post = rec[rec["age_at_comp"] > cutoff]
            pb_post = post.groupby("athlete_id")["time_raw"].min()
            win = rec[(rec["age_at_comp"] > cutoff)
                      & (rec["age_at_comp"] <= cutoff + FOLLOWUP_YEARS)]
            pb_win = win.groupby("athlete_id")["time_raw"].min()
            pre = rec[rec["age_at_comp"] <= cutoff]
            pb_pre = pre.groupby("athlete_id")["time_raw"].min()

            for sex_code, sex_label in SEXES:
                ids = gl[(gl[f"cutoff_{cutoff}_group"] == "A")
                         & (gl["sex"] == sex_code)]["athlete_id"].tolist()
                f = build(ids, cutoff)
                cols = cols_for(f, VARIANT)
                targets = {
                    "pb_lifetime": gl.set_index("athlete_id")["lifetime_pb"].reindex(f.index),
                    "pb_post_cutoff": pb_post.reindex(f.index),
                    "improvement_2y": (pb_pre.reindex(f.index) - pb_win.reindex(f.index)),
                }
                for tname, tvals in targets.items():
                    sub = f.loc[tvals.notna()]
                    y = tvals.dropna().values.astype(np.float64)
                    if len(y) < 40:
                        print(f"  {sex_label[:1]}{cutoff} {tname}: n={len(y)} < 40, skipped")
                        continue
                    for mname in MODELS:
                        t0 = time.time()
                        r = random_cv(models[mname], sub[cols].values.astype(np.float64),
                                      y, n_folds, n_rep, seed, n_jobs=n_jobs)
                        rows.append(dict(block="A", cutoff_age=cutoff, sex=sex_label,
                                         outcome=tname, model=mname, n=len(y),
                                         pct_of_group_a=round(100 * len(y) / len(f), 1),
                                         R2_mean=r["R2_mean"], R2_ci_lo=r["R2_ci_lo"],
                                         R2_ci_hi=r["R2_ci_hi"], RMSE_mean=r["RMSE_mean"],
                                         n_repeats=n_rep))
                        print(f"  {sex_label[:1]}{cutoff} {tname:<16}{mname:<18}"
                              f"n={len(y):<5} R2={r['R2_mean']:.4f} "
                              f"[{r['R2_ci_lo']:.3f}, {r['R2_ci_hi']:.3f}]  "
                              f"{time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(tdir / "r4_post_cutoff_outcome.csv", index=False)
        print(f"\nSaved {tdir/'r4_post_cutoff_outcome.csv'}")

    # =====================================================================
    # B.  balanced panel
    # =====================================================================
    if "B" in only:
        print("\n" + "=" * 74)
        print("B.  Reviewer 4, comment 2: balanced panel across cutoffs 16, 17, 18")
        print("=" * 74)
        rows = []
        for sex_code, sex_label in SEXES:
            d = gl[gl["sex"] == sex_code]
            keep = d[(d["cutoff_16_group"] == "A") & (d["cutoff_17_group"] == "A")
                     & (d["cutoff_18_group"] == "A")]["athlete_id"].tolist()
            print(f"  {sex_label}: Group A at all three cutoffs -> n = {len(keep)} "
                  f"(of {int((d['cutoff_16_group']=='A').sum())} / "
                  f"{int((d['cutoff_17_group']=='A').sum())} / "
                  f"{int((d['cutoff_18_group']=='A').sum())})")
            if len(keep) < 40:
                print("    too few for cross-validation, reported as such")
                continue
            for cutoff in CUTOFFS:
                f = build(keep, cutoff)
                f["y"] = gl.set_index("athlete_id")["lifetime_pb"].reindex(f.index)
                f = f.dropna(subset=["y"])
                cols = cols_for(f, VARIANT)
                for mname in MODELS:
                    t0 = time.time()
                    r = random_cv(models[mname], f[cols].values.astype(np.float64),
                                  f["y"].values.astype(np.float64),
                                  n_folds, n_rep, seed, n_jobs=n_jobs)
                    rows.append(dict(block="B", cutoff_age=cutoff, sex=sex_label,
                                     model=mname, n=len(f),
                                     R2_mean=r["R2_mean"], R2_ci_lo=r["R2_ci_lo"],
                                     R2_ci_hi=r["R2_ci_hi"], RMSE_mean=r["RMSE_mean"],
                                     ci_width=r["R2_ci_hi"] - r["R2_ci_lo"],
                                     n_repeats=n_rep))
                    print(f"    cutoff {cutoff} {mname:<18} n={len(f):<4} "
                          f"R2={r['R2_mean']:.4f} [{r['R2_ci_lo']:.3f}, "
                          f"{r['R2_ci_hi']:.3f}]  width={r['R2_ci_hi']-r['R2_ci_lo']:.3f}"
                          f"  {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(tdir / "r4_balanced_panel.csv", index=False)
        print(f"\nSaved {tdir/'r4_balanced_panel.csv'}")

    # =====================================================================
    # C.  paired anthropometric delta-R2
    # =====================================================================
    if "C" in only:
        print("\n" + "=" * 74)
        print("C.  Reviewer 4, comment 6: paired CI for the anthropometric contrast")
        print("=" * 74)
        rows = []
        for cutoff in CUTOFFS:
            for sex_code, sex_label in SEXES:
                d = gl[(gl[f"cutoff_{cutoff}_group"] == "A") & (gl["sex"] == sex_code)]
                ids = d[d["has_anthro"]]["athlete_id"].tolist()
                f = build(ids, cutoff)
                for c in ["height_cm", "weight_kg"]:
                    f[c] = d.set_index("athlete_id")[c].reindex(f.index)
                f["bmi"] = f["weight_kg"] / (f["height_cm"] / 100.0) ** 2
                f["height_missing"] = f["height_cm"].isna().astype(float)
                f["weight_missing"] = f["weight_kg"].isna().astype(float)
                f["y"] = gl.set_index("athlete_id")["lifetime_pb"].reindex(f.index)
                f = f.dropna(subset=["y"])
                y = f["y"].values.astype(np.float64)

                spec = {
                    "no_anthro": cols_for(f, "full_no_anthro"),
                    "with_anthro": cols_for(f, "full_no_anthro")
                                   + ["height_cm", "weight_kg", "bmi",
                                      "height_missing", "weight_missing"],
                }
                X = {k: f[v].values.astype(np.float64) for k, v in spec.items()}

                def _rep(s):
                    kf = KFold(n_splits=n_folds, shuffle=True, random_state=s)
                    out = []
                    for fi, (tr, te) in enumerate(kf.split(y)):
                        for k, Xk in X.items():
                            for mname in MODELS:
                                m = clone(models[mname])
                                m.fit(Xk[tr], y[tr])
                                out.append((s, fi, k, mname,
                                            r2_score(y[te], m.predict(Xk[te]))))
                    return out

                t0 = time.time()
                chunks = Parallel(n_jobs=n_jobs)(
                    delayed(_rep)(seed + i) for i in range(n_rep))
                fr = pd.DataFrame([r for c in chunks for r in c],
                                  columns=["seed", "fold", "spec", "model", "r2"])
                wide = fr.pivot_table(index=["seed", "fold"],
                                      columns=["spec", "model"], values="r2")
                for mname in MODELS:
                    a, b = ("with_anthro", mname), ("no_anthro", mname)
                    diff = (wide[a] - wide[b]).values
                    rows.append(dict(
                        block="C", cutoff_age=cutoff, sex=sex_label, model=mname,
                        n=len(y),
                        r2_with=wide[a].mean(), r2_without=wide[b].mean(),
                        delta_mean=diff.mean(),
                        delta_ci_lo=np.percentile(diff, 2.5),
                        delta_ci_hi=np.percentile(diff, 97.5),
                        excludes_zero=bool(np.percentile(diff, 2.5) > 0
                                           or np.percentile(diff, 97.5) < 0),
                        marginal_ci_with=f"[{np.percentile(wide[a],2.5):.3f}, "
                                         f"{np.percentile(wide[a],97.5):.3f}]",
                        marginal_ci_without=f"[{np.percentile(wide[b],2.5):.3f}, "
                                            f"{np.percentile(wide[b],97.5):.3f}]",
                        n_repeats=n_rep))
                    r = rows[-1]
                    print(f"  {sex_label[:1]}{cutoff} {mname:<18} n={len(y):<5} "
                          f"delta={r['delta_mean']:+.4f} "
                          f"[{r['delta_ci_lo']:+.4f}, {r['delta_ci_hi']:+.4f}]  "
                          f"excl0={r['excludes_zero']}  "
                          f"(marginal {r['marginal_ci_without']} vs "
                          f"{r['marginal_ci_with']})  {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(tdir / "r4_anthro_paired_delta.csv", index=False)
        print(f"\nSaved {tdir/'r4_anthro_paired_delta.csv'}")

    print("\nNote for the response letter: random versus province-disjoint validation "
          "is not\npaired here. KFold and GroupKFold partition the data differently by "
          "construction,\nso identical folds do not exist for that contrast.")


if __name__ == "__main__":
    main()
