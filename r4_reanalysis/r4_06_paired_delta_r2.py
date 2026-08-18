"""
Reviewer 4, major comment #6 -- confidence intervals for R2 DIFFERENCES from
paired resampling on identical folds.
============================================================================
The published intervals are percentiles of each model's own marginal fold-R2
distribution. Comparing two such intervals for overlap is not a test of their
difference, which is R4's objection and also the reason paragraph 57's
"non-overlapping CIs" argument has to go.

Here every competing specification is fitted on THE SAME (seed, fold) split, the
difference is taken within the fold, and the interval is the percentile of the
2500 paired differences. Because the fold-to-fold variance is shared, these
intervals are much narrower than the difference of the marginal intervals.

Specifications compared per cell (cutoff x sex), all on Group A:
    trajectory_only / traj_wind / full_no_anthro   x   Ridge / GradientBoosting

Output: results/tables/r4_paired_delta_r2.csv       (all pairwise contrasts)
        results/tables/r4_paired_fold_r2.csv        (per-fold R2, for reuse)

Usage:  python3 r4_06_paired_delta_r2.py [--repeats N] [--cells M16,F16,...]
"""
import sys
import time
from itertools import combinations
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent.parent   # repo root; this script lives in r4_reanalysis/
sys.path.insert(0, str(HERE / "src"))

from utils.io import load_config, read_parquet      # noqa: E402
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


VARIANTS = ["trajectory_only", "traj_wind", "full_no_anthro"]
MODELS = ["Ridge", "GradientBoosting"]
CELLS = [(16, "M", "Male"), (16, "F", "Female"),
         (17, "M", "Male"), (17, "F", "Female"),
         (18, "M", "Male"), (18, "F", "Female")]


def arg(flag: str, default):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


def paired_fold_r2(spec_X: dict, y: np.ndarray, models: dict,
                   n_folds: int, n_repeats: int, seed: int, n_jobs: int) -> pd.DataFrame:
    """
    For every (seed, fold) split, fit every (variant, model) specification on the
    SAME training rows and score on the SAME test rows. Returns tidy per-fold R2.
    """
    def _one_repeat(s: int):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=s)
        out = []
        for fi, (tr, te) in enumerate(kf.split(y)):
            for variant, X in spec_X.items():
                for mname in MODELS:
                    m = clone(models[mname])
                    m.fit(X[tr], y[tr])
                    out.append((s, fi, variant, mname,
                                r2_score(y[te], m.predict(X[te]))))
        return out

    seeds = [seed + i for i in range(n_repeats)]
    chunks = Parallel(n_jobs=n_jobs)(delayed(_one_repeat)(s) for s in seeds)
    flat = [r for c in chunks for r in c]
    return pd.DataFrame(flat, columns=["seed", "fold", "variant", "model", "r2"])


def main() -> None:
    require_full_dataset()
    n_repeats = int(arg("--repeats", 500))
    only = arg("--cells", None)
    cells = CELLS
    if only:
        keep = set(only.split(","))
        cells = [c for c in CELLS if f"{c[1]}{c[0]}" in keep]

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

    fold_frames, contrast_rows = [], []

    for cutoff, sex_code, sex_label in cells:
        t0 = time.time()
        a_ids = group_labels[
            (group_labels[f"cutoff_{cutoff}_group"] == "A")
            & (group_labels["sex"] == sex_code)
        ]["athlete_id"].tolist()

        feats = s11.build_features_for_group_a(
            a_ids, rec_100m, athletes, cutoff,
            config["features"]["min_records_for_slope"],
        )
        pb_map = group_labels.set_index("athlete_id")["lifetime_pb"]
        feats["lifetime_pb"] = pb_map.loc[feats.index]
        feats = feats.dropna(subset=["lifetime_pb"])
        y = feats["lifetime_pb"].values.astype(np.float64)

        spec_X = {}
        for v in VARIANTS:
            cols = [c for c in s11.get_variant_cols(registry, v) if c in feats.columns]
            spec_X[v] = feats[cols].values.astype(np.float64)

        fr = paired_fold_r2(spec_X, y, models, n_folds, n_repeats, seed, n_jobs)
        fr["cutoff_age"], fr["sex"], fr["n"] = cutoff, sex_label, len(y)
        fold_frames.append(fr)

        # ---- marginal means (must equal the published table) ----------------
        marg = fr.groupby(["variant", "model"])["r2"].agg(
            mean="mean",
            ci_lo=lambda s: np.percentile(s, 2.5),
            ci_hi=lambda s: np.percentile(s, 97.5),
        )

        # ---- paired contrasts ------------------------------------------------
        wide = fr.pivot_table(index=["seed", "fold"], columns=["variant", "model"],
                              values="r2")
        specs = list(wide.columns)
        for a, b in combinations(specs, 2):
            d = (wide[a] - wide[b]).values
            contrast_rows.append({
                "cutoff_age": cutoff, "sex": sex_label, "n": len(y),
                "spec_a": f"{a[0]}|{a[1]}", "spec_b": f"{b[0]}|{b[1]}",
                "r2_a": marg.loc[a, "mean"], "r2_b": marg.loc[b, "mean"],
                "delta_mean": d.mean(),
                "delta_ci_lo": np.percentile(d, 2.5),
                "delta_ci_hi": np.percentile(d, 97.5),
                "pct_folds_a_better": float((d > 0).mean() * 100),
                "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0),
                # width of the naive comparison, for contrast
                "naive_ci_a": f"[{marg.loc[a,'ci_lo']:.3f}, {marg.loc[a,'ci_hi']:.3f}]",
                "naive_ci_b": f"[{marg.loc[b,'ci_lo']:.3f}, {marg.loc[b,'ci_hi']:.3f}]",
                "n_repeats": n_repeats,
            })

        print(f"cutoff {cutoff} {sex_label} (n={len(y)}) "
              f"{time.time()-t0:.0f}s  "
              f"traj_wind|GB={marg.loc[('traj_wind','GradientBoosting'),'mean']:.6f}",
              flush=True)

    tdir = HERE / "results/tables"
    pd.concat(fold_frames).to_csv(tdir / "r4_paired_fold_r2.csv", index=False)
    out = pd.DataFrame(contrast_rows)
    out.to_csv(tdir / "r4_paired_delta_r2.csv", index=False)
    print(f"\nSaved {tdir/'r4_paired_delta_r2.csv'}  ({len(out)} contrasts)")

    key = out[out.spec_a.str.startswith("traj_wind")
              & out.spec_b.str.startswith("traj_wind")]
    if len(key):
        print("\nRidge vs GradientBoosting on traj_wind (paired):")
        for _, r in key.iterrows():
            print(f"  {r.sex:<7}{r.cutoff_age}  {r.spec_a.split('|')[1]:<17}"
                  f"minus {r.spec_b.split('|')[1]:<17}"
                  f"delta={r.delta_mean:+.4f} [{r.delta_ci_lo:+.4f}, {r.delta_ci_hi:+.4f}]"
                  f"  excludes 0: {r.excludes_zero}")


if __name__ == "__main__":
    main()
