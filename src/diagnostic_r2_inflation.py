"""
Diagnostic Analysis: R² Inflation Investigation
================================================
Investigates why R² at cutoff=16 is already 0.87 in CAA data
vs 0.04 in WA data.

Three diagnostic modules:
  1. Career overlap rate per cutoff age
  2. PB variance decomposition (SD/range vs WA)
  3. Stratified A/B test at EVERY cutoff age (16–26):
       Group A = athletes with post-cutoff records ("true prediction")
       Group B = athletes whose career ends at or before cutoff ("memory")

Run:  python src/diagnostic_r2_inflation.py
Requires: data/interim/cleaned_records.parquet + data/interim/parsed_athletes.parquet
          (outputs from s01–s02 in the main pipeline)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.base import clone

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.io import load_config, read_parquet, write_parquet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("diagnostic.log")],
)
logger = logging.getLogger(__name__)

# WA reference values.
# PB SDs are the overall (all-athlete) PB SDs from the World Athletics analysis
# table1_descriptive_stats.md. The R2 series is the 500-repeat cutoff-curve sweep
# (sprint_pipeline/raw_result/cutoff_curve_500boot_{male,female}.csv), which is the
# only WA analysis spanning cutoffs 16-26. The values previously hard-coded here at
# cutoffs 16/17/19/21/24/26 were hand-entered approximations and were wrong
# (male 17 was 0.200 vs 0.270 true; male 19 was 0.500 vs 0.470 true).
WA_REF = {
    "male_pb_mean": 10.462,
    "male_pb_sd": 0.200,
    "female_pb_mean": 11.566,
    "female_pb_sd": 0.325,
    "male_r2": {16: 0.0403, 17: 0.2698, 18: 0.4306, 19: 0.4700, 20: 0.5576,
                21: 0.6624, 22: 0.7563, 23: 0.8250, 24: 0.8787, 25: 0.9119,
                26: 0.9408},
    "female_r2": {16: 0.1980, 17: 0.2614, 18: 0.3553, 19: 0.4136, 20: 0.5074,
                  21: 0.6188, 22: 0.7246, 23: 0.7842, 24: 0.8346, 25: 0.8728,
                  26: 0.8982},
}

CUTOFF_AGES = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

# Quick CV settings for diagnostic (not final analysis)
N_FOLDS = 5
N_REPEATS = 10  # Fast; bump to 100+ for publication
GB_PARAMS = dict(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)


# ---------------------------------------------------------------------------
# Helper: quick R² via repeated k-fold
# ---------------------------------------------------------------------------
def quick_r2(X: np.ndarray, y: np.ndarray, n_repeats: int = N_REPEATS) -> dict:
    """Lightweight repeated CV → mean R² and 95% CI."""
    if len(X) < 30:
        return {"r2_mean": np.nan, "r2_lo": np.nan, "r2_hi": np.nan, "n": len(X)}

    fold_r2s = []
    for seed in range(n_repeats):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr, te in kf.split(X):
            m = GradientBoostingRegressor(**GB_PARAMS)
            m.fit(X[tr], y[tr])
            ss_res = np.sum((y[te] - m.predict(X[te])) ** 2)
            ss_tot = np.sum((y[te] - y[te].mean()) ** 2)
            fold_r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)

    arr = np.array(fold_r2s)
    return {
        "r2_mean": np.mean(arr),
        "r2_lo": np.percentile(arr, 2.5),
        "r2_hi": np.percentile(arr, 97.5),
        "n": len(X),
    }


# ---------------------------------------------------------------------------
# Helper: build minimal feature vector for one cutoff
# ---------------------------------------------------------------------------
def build_minimal_features(records_100m: pd.DataFrame, cutoff_age: float) -> pd.DataFrame:
    """
    Build a lean feature matrix (best_time, mean_time, sd_time, n_records,
    improvement_slope_simple, age_first) from pre-cutoff 100m records.
    Just enough for diagnostic R² — NOT the full feature set.
    """
    pre = records_100m[records_100m["age_at_comp"] <= cutoff_age].copy()

    # Require ≥ 2 pre-cutoff records
    counts = pre.groupby("athlete_id").size()
    valid_ids = counts[counts >= 2].index
    pre = pre[pre["athlete_id"].isin(valid_ids)]

    def _feats(g):
        times = g["time_raw"].values
        ages = g["age_at_comp"].values
        span = ages.max() - ages.min()
        slope = np.nan
        if len(times) >= 3 and span > 0:
            # simple OLS slope (fast, good enough for diagnostic)
            slope = np.polyfit(ages, times, 1)[0]
        elif len(times) >= 2 and span > 0:
            slope = (times[-1] - times[0]) / span

        return pd.Series({
            "best_time": times.min(),
            "mean_time": times.mean(),
            "sd_time": times.std() if len(times) > 1 else 0.0,
            "n_records": len(times),
            "slope": slope if not np.isnan(slope) else 0.0,
            "age_first": ages.min(),
        })

    feats = pre.groupby("athlete_id").apply(_feats)
    return feats


# ===========================================================================
# DIAGNOSTIC 1: Career overlap rate
# ===========================================================================
def diagnostic_1_career_overlap(records_100m: pd.DataFrame, lifetime_pb: pd.Series):
    """
    For each cutoff age, what % of modeling-sample athletes already achieved
    their lifetime PB in the pre-cutoff window?
    """
    logger.info("=" * 70)
    logger.info("DIAGNOSTIC 1: Career Overlap Rate")
    logger.info("=" * 70)

    rows = []
    for cutoff in CUTOFF_AGES:
        pre = records_100m[records_100m["age_at_comp"] <= cutoff]
        counts = pre.groupby("athlete_id").size()
        valid_ids = counts[counts >= 2].index

        pre_best = pre[pre["athlete_id"].isin(valid_ids)].groupby("athlete_id")["time_raw"].min()
        lt_pb = lifetime_pb.loc[valid_ids]

        # PB already achieved = pre-cutoff best == lifetime PB
        overlap = (np.abs(pre_best - lt_pb) < 0.001).mean()

        # Also: % whose career ends at or before cutoff (no post-cutoff records)
        all_last_age = records_100m.groupby("athlete_id")["age_at_comp"].max()
        career_ended = (all_last_age.loc[valid_ids] <= cutoff).mean()

        rows.append({
            "cutoff_age": cutoff,
            "n_athletes": len(valid_ids),
            "pct_pb_already_achieved": round(overlap * 100, 1),
            "pct_career_ended": round(career_ended * 100, 1),
        })

        logger.info(
            f"  Cutoff {cutoff}: n={len(valid_ids):>6,}  |  "
            f"PB already achieved: {overlap*100:5.1f}%  |  "
            f"Career ended: {career_ended*100:5.1f}%"
        )

    df = pd.DataFrame(rows)
    logger.info("\n" + df.to_string(index=False))
    return df


# ===========================================================================
# DIAGNOSTIC 2: PB variance decomposition
# ===========================================================================
def diagnostic_2_pb_variance(records_100m: pd.DataFrame, athletes: pd.DataFrame,
                             lifetime_pb: pd.Series):
    """
    Compare PB distribution (SD, range, IQR) with WA reference values.
    Break down by sex and by ability stratum.
    """
    logger.info("=" * 70)
    logger.info("DIAGNOSTIC 2: PB Variance Decomposition")
    logger.info("=" * 70)

    # Merge sex
    pb_df = lifetime_pb.reset_index()
    pb_df.columns = ["athlete_id", "lifetime_pb"]
    pb_df = pb_df.merge(athletes[["athlete_id", "sex"]], on="athlete_id", how="left")

    rows = []
    for sex, label in [("M", "Male"), ("F", "Female")]:
        vals = pb_df.loc[pb_df["sex"] == sex, "lifetime_pb"].dropna()
        wa_sd = WA_REF[f"{label.lower()}_pb_sd"]
        wa_mean = WA_REF[f"{label.lower()}_pb_mean"]

        row = {
            "sex": label,
            "n": len(vals),
            "mean": round(vals.mean(), 3),
            "sd": round(vals.std(), 3),
            "min": round(vals.min(), 3),
            "q25": round(vals.quantile(0.25), 3),
            "median": round(vals.median(), 3),
            "q75": round(vals.quantile(0.75), 3),
            "max": round(vals.max(), 3),
            "iqr": round(vals.quantile(0.75) - vals.quantile(0.25), 3),
            "range": round(vals.max() - vals.min(), 3),
            "wa_mean": wa_mean,
            "wa_sd": wa_sd,
            "sd_ratio_vs_wa": round(vals.std() / wa_sd, 2),
        }
        rows.append(row)
        logger.info(
            f"  {label}: n={len(vals):,}  mean={vals.mean():.3f}  SD={vals.std():.3f}  "
            f"range=[{vals.min():.2f}, {vals.max():.2f}]  "
            f"WA_SD={wa_sd}  SD_ratio={vals.std()/wa_sd:.2f}x"
        )

    df = pd.DataFrame(rows)
    logger.info("\n" + df.to_string(index=False))
    return df


# ===========================================================================
# DIAGNOSTIC 3: Stratified A/B test at every cutoff age
# ===========================================================================
def diagnostic_3_stratified_r2(records_100m: pd.DataFrame, athletes: pd.DataFrame,
                               lifetime_pb: pd.Series):
    """
    At EACH cutoff age, split into:
      Group A: athletes with at least 1 post-cutoff record ("true prediction")
      Group B: athletes whose last record is at or before cutoff ("memory/trivial")
    Run quick GBM CV on each group separately + combined.

    Extended: also try cutoffs 23, 25 as requested.
    """
    logger.info("=" * 70)
    logger.info("DIAGNOSTIC 3: Stratified A/B R² at Every Cutoff Age")
    logger.info("=" * 70)

    # Pre-compute: last competition age per athlete (across ALL events/records)
    last_age_all = records_100m.groupby("athlete_id")["age_at_comp"].max()

    rows = []
    for sex, sex_label in [("M", "Male"), ("F", "Female")]:
        sex_ids = set(athletes.loc[athletes["sex"] == sex, "athlete_id"])
        rec_sex = records_100m[records_100m["athlete_id"].isin(sex_ids)]

        for cutoff in CUTOFF_AGES:
            logger.info(f"  {sex_label} cutoff={cutoff} ...")

            # Build features for this cutoff
            feats = build_minimal_features(rec_sex, cutoff)
            if len(feats) < 50:
                logger.warning(f"    Only {len(feats)} athletes, skipping")
                continue

            # Add target
            feats["lifetime_pb"] = lifetime_pb.loc[feats.index]
            feats = feats.dropna(subset=["lifetime_pb"])

            # Split A vs B
            last_ages = last_age_all.loc[feats.index]
            mask_A = last_ages > cutoff    # Has post-cutoff records
            mask_B = last_ages <= cutoff   # Career ended at/before cutoff

            feature_cols = ["best_time", "mean_time", "sd_time", "n_records", "slope", "age_first"]

            # --- Combined (all) ---
            X_all = feats[feature_cols].values
            y_all = feats["lifetime_pb"].values
            res_all = quick_r2(X_all, y_all)

            # --- Group A: true prediction ---
            X_A = feats.loc[mask_A, feature_cols].values
            y_A = feats.loc[mask_A, "lifetime_pb"].values
            res_A = quick_r2(X_A, y_A)

            # --- Group B: trivial/memory ---
            X_B = feats.loc[mask_B, feature_cols].values
            y_B = feats.loc[mask_B, "lifetime_pb"].values
            res_B = quick_r2(X_B, y_B)

            # PB already achieved rate for Group A specifically
            pre_best_A = rec_sex[
                (rec_sex["athlete_id"].isin(feats.index[mask_A])) &
                (rec_sex["age_at_comp"] <= cutoff)
            ].groupby("athlete_id")["time_raw"].min()
            lt_pb_A = lifetime_pb.loc[pre_best_A.index]
            pct_pb_pre_A = (np.abs(pre_best_A - lt_pb_A) < 0.001).mean() * 100

            # WA reference
            wa_key = f"{sex_label.lower()}_r2"
            wa_r2 = WA_REF[wa_key].get(cutoff, np.nan)

            row = {
                "sex": sex_label,
                "cutoff": cutoff,
                "n_all": res_all["n"],
                "n_A": res_A["n"],
                "n_B": res_B["n"],
                "pct_A": round(res_A["n"] / res_all["n"] * 100, 1) if res_all["n"] > 0 else 0,
                "r2_all": round(res_all["r2_mean"], 4),
                "r2_A_true_pred": round(res_A["r2_mean"], 4),
                "r2_A_ci_lo": round(res_A["r2_lo"], 4),
                "r2_A_ci_hi": round(res_A["r2_hi"], 4),
                "r2_B_trivial": round(res_B["r2_mean"], 4),
                "pct_pb_pre_in_A": round(pct_pb_pre_A, 1),
                "pb_sd_all": round(feats["lifetime_pb"].std(), 3),
                "pb_sd_A": round(feats.loc[mask_A, "lifetime_pb"].std(), 3) if mask_A.sum() > 1 else np.nan,
                "pb_sd_B": round(feats.loc[mask_B, "lifetime_pb"].std(), 3) if mask_B.sum() > 1 else np.nan,
                "wa_r2": wa_r2,
            }
            rows.append(row)

            logger.info(
                f"    ALL: n={res_all['n']:>5} R²={res_all['r2_mean']:.3f}  |  "
                f"A(true): n={res_A['n']:>5} R²={res_A['r2_mean']:.3f} [{res_A['r2_lo']:.3f},{res_A['r2_hi']:.3f}]  |  "
                f"B(mem):  n={res_B['n']:>5} R²={res_B['r2_mean']:.3f}  |  "
                f"WA: {wa_r2 if not np.isnan(wa_r2) else 'n/a'}"
            )

    df = pd.DataFrame(rows)
    return df


# ===========================================================================
# Main
# ===========================================================================
def main():
    config = load_config()
    interim = Path(config["paths"]["interim"])
    results_dir = Path(config["paths"]["results"]) / "diagnostic"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    records = read_parquet(interim / "cleaned_records.parquet")
    athletes = read_parquet(interim / "parsed_athletes.parquet")

    # Filter to 100m only
    records_100m = records[records["event"] == "100m"].copy()
    logger.info(f"100m records: {len(records_100m):,} from {records_100m['athlete_id'].nunique():,} athletes")

    # Compute lifetime PB (across ALL records, no cutoff — this is the target)
    lifetime_pb = records_100m.groupby("athlete_id")["time_raw"].min()
    lifetime_pb.name = "lifetime_pb"
    logger.info(f"Lifetime PBs computed for {len(lifetime_pb):,} athletes")

    # --- Run diagnostics ---
    df1 = diagnostic_1_career_overlap(records_100m, lifetime_pb)
    df2 = diagnostic_2_pb_variance(records_100m, athletes, lifetime_pb)
    df3 = diagnostic_3_stratified_r2(records_100m, athletes, lifetime_pb)

    # --- Save ---
    df1.to_csv(results_dir / "diag1_career_overlap.csv", index=False)
    df2.to_csv(results_dir / "diag2_pb_variance.csv", index=False)
    df3.to_csv(results_dir / "diag3_stratified_r2.csv", index=False)

    # --- Print summary table ---
    logger.info("\n" + "=" * 90)
    logger.info("SUMMARY: Group A (True Prediction) R² vs WA Reference")
    logger.info("=" * 90)

    summary_cols = ["sex", "cutoff", "n_A", "pct_A", "r2_A_true_pred", "r2_A_ci_lo",
                    "r2_A_ci_hi", "r2_all", "r2_B_trivial", "wa_r2", "pb_sd_A"]
    if len(df3) > 0:
        logger.info("\n" + df3[summary_cols].to_string(index=False))

    # --- Key decision metrics ---
    logger.info("\n" + "=" * 90)
    logger.info("KEY DECISION METRICS")
    logger.info("=" * 90)

    for sex in ["Male", "Female"]:
        sub = df3[df3["sex"] == sex]
        if len(sub) == 0:
            continue
        logger.info(f"\n  {sex}:")
        for _, row in sub.iterrows():
            delta = row["r2_all"] - row["r2_A_true_pred"]
            logger.info(
                f"    Cutoff {int(row['cutoff']):>2}: "
                f"R²_all={row['r2_all']:.3f}  R²_A={row['r2_A_true_pred']:.3f}  "
                f"Δ={delta:+.3f}  "
                f"Group_A={row['pct_A']:.0f}%  "
                f"PB_SD_A={row['pb_sd_A']}  "
                f"WA={row['wa_r2'] if not pd.isna(row['wa_r2']) else 'n/a'}"
            )

    logger.info(f"\nResults saved to {results_dir}/")
    logger.info("Done.")


if __name__ == "__main__":
    main()
