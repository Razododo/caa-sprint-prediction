"""
Step 5: Build feature matrices at each cutoff age.

Input:  data/interim/cleaned_records.parquet + cleaned_athletes.parquet
Output: data/processed/features_cutoff_{age}.parquet for each cutoff age
        data/processed/modeling_sample.parquet (metadata: who qualifies at each cutoff)
        data/processed/anthro_subsample.parquet (athletes with valid height/weight)

CRITICAL — NO DATA LEAKAGE:
    At cutoff age X, only records where age_at_comp ≤ X enter feature computation.
    Target = lifetime PB across ALL records (including post-cutoff).
"""
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.io import load_config, ensure_dirs, read_parquet, write_parquet
from utils.features import (
    compute_trajectory_raw, compute_trajectory_wc, compute_trajectory_dynamics,
    compute_career_structure, compute_round_performance, compute_wind_features,
    compute_anthropometric, compute_era_features,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_features_for_athlete(
    recs: pd.DataFrame,
    athlete_row: pd.Series,
    min_for_slope: int,
) -> dict:
    """Build all feature groups for a single athlete's pre-cutoff records."""
    feats: dict = {}
    feats.update(compute_trajectory_raw(recs))
    feats.update(compute_trajectory_wc(recs))
    feats.update(compute_trajectory_dynamics(recs, min_for_slope))
    feats.update(compute_career_structure(recs))
    feats.update(compute_round_performance(recs))
    feats.update(compute_wind_features(recs))
    feats.update(compute_anthropometric(athlete_row.get("height_cm"), athlete_row.get("weight_kg")))
    feats.update(compute_era_features(recs, athlete_row.get("dob")))
    feats["athlete_id"] = athlete_row.name
    feats["sex"] = athlete_row["sex"]
    return feats


def main() -> None:
    config = load_config()
    ensure_dirs(config)

    interim = Path(config["paths"]["interim"])
    processed = Path(config["paths"]["processed"])

    df_ath = read_parquet(interim / "cleaned_athletes.parquet")
    df_rec = read_parquet(interim / "cleaned_records.parquet")

    cutoff_ages = config["features"]["cutoff_ages"]
    min_100m = config["inclusion"]["min_100m_records"]
    min_for_slope = config["features"]["min_records_for_slope"]
    h_range = config["inclusion"]["height_range"]
    w_range = config["inclusion"]["weight_range"]

    # Only 100m records for feature building
    rec_100m = df_rec[df_rec["event"] == "100m"].copy()
    logger.info(f"100m records for feature engineering: {len(rec_100m):,}")

    # Lifetime PB: minimum raw time across ALL 100m records (no cutoff restriction)
    lifetime_pb = rec_100m.groupby("athlete_id")["time_raw"].min()
    lifetime_pb.name = "lifetime_pb"

    # Athlete metadata indexed by athlete_id
    df_ath = df_ath.set_index("athlete_id")

    # Identify anthro subsample
    has_height = df_ath["height_cm"].notna() & df_ath["height_cm"].between(h_range[0], h_range[1])
    has_weight = df_ath["weight_kg"].notna() & df_ath["weight_kg"].between(w_range[0], w_range[1])
    anthro_ids = set(df_ath.index[has_height & has_weight])
    logger.info(f"Anthro subsample (valid H+W in range): {len(anthro_ids):,}")

    summary_rows = []

    for cutoff in cutoff_ages:
        logger.info(f"\n{'='*50}")
        logger.info(f"Building features for cutoff age {cutoff}")

        # Filter to pre-cutoff records
        pre = rec_100m[rec_100m["age_at_comp"] <= cutoff].copy()

        # Athletes with ≥ min_100m records in the window
        ath_counts = pre.groupby("athlete_id").size()
        qualifying = set(ath_counts[ath_counts >= min_100m].index)

        # Also need lifetime PB
        qualifying = qualifying & set(lifetime_pb.index)

        logger.info(f"  Pre-cutoff records: {len(pre):,}, qualifying athletes: {len(qualifying):,}")

        if not qualifying:
            logger.warning(f"  No qualifying athletes at cutoff {cutoff}")
            continue

        pre_qual = pre[pre["athlete_id"].isin(qualifying)]
        grouped = pre_qual.groupby("athlete_id")

        feat_rows = []
        for aid, grp in tqdm(grouped, desc=f"  Cutoff {cutoff}", leave=False):
            if aid not in df_ath.index:
                continue
            row = build_features_for_athlete(grp, df_ath.loc[aid], min_for_slope)
            row["lifetime_pb"] = lifetime_pb[aid]
            row["has_anthro"] = int(aid in anthro_ids)
            feat_rows.append(row)

        df_feat = pd.DataFrame(feat_rows)
        logger.info(f"  Feature matrix: {df_feat.shape[0]} athletes × {df_feat.shape[1]} features")

        n_m = (df_feat["sex"] == "M").sum()
        n_f = (df_feat["sex"] == "F").sum()
        n_anthro = df_feat["has_anthro"].sum()
        logger.info(f"  Male: {n_m}, Female: {n_f}, with anthro: {n_anthro}")

        write_parquet(df_feat, processed / f"features_cutoff_{cutoff}.parquet")

        summary_rows.append({
            "cutoff_age": cutoff,
            "n_athletes": len(df_feat),
            "n_male": n_m,
            "n_female": n_f,
            "n_anthro": n_anthro,
        })

    # Save modeling sample summary
    df_summary = pd.DataFrame(summary_rows)
    write_parquet(df_summary, processed / "modeling_sample_summary.parquet")
    logger.info(f"\nModeling sample summary:\n{df_summary.to_string(index=False)}")


if __name__ == "__main__":
    main()
