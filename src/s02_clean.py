"""
Step 2: Data cleaning, validation, and deduplication.

Input:  data/interim/parsed_athletes.parquet + all_records_raw.parquet
Output: data/interim/cleaned_records.parquet (updated records with age, round_type, etc.)
        data/interim/cleaned_athletes.parquet (athletes after DOB filter)
        data/interim/cleaning_log.json (summary of records removed at each step)

Cleaning steps:
1. Remove records with invalid times (NaN, ≤0, or outside physiological range)
1b. Remove 100 m hurdles records mis-filed under event == "100m"
2. Remove athletes without DOB
3. Compute age_at_comp = (competition_date - dob) / 365.25
4. Remove records where age < 12 or age > 45
5. Classify round_type from event_full (预赛/半决赛/决赛)
6. Flag indoor events
7. Deduplicate: same athlete + event + date + time → keep one
8. Log cleaning summary
"""
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from utils.io import load_config, ensure_dirs, read_parquet, write_parquet, save_results_meta

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def classify_round(event_full: str) -> str:
    """Classify round type from full event description."""
    if not isinstance(event_full, str):
        return "unknown"
    if "决赛" in event_full and "半决赛" not in event_full:
        return "final"
    if "半决赛" in event_full:
        return "semi"
    if "预赛" in event_full:
        return "heats"
    return "unknown"


def is_indoor(event: str, event_full: str, comp_name: str) -> bool:
    """Determine if event is indoor."""
    if event == "60m":
        return True
    text = f"{event_full} {comp_name}".lower()
    return "室内" in text or "indoor" in text or "ind." in text


def main() -> None:
    config = load_config()
    ensure_dirs(config)
    inc = config["inclusion"]

    interim = Path(config["paths"]["interim"])
    df_ath = read_parquet(interim / "parsed_athletes.parquet")
    df_rec = read_parquet(interim / "all_records_raw.parquet")

    cleaning_log: dict[str, int] = {"records_initial": len(df_rec), "athletes_initial": len(df_ath)}

    # --- Step 1: Remove invalid times ---
    valid_mask = df_rec["time_raw"].notna() & (df_rec["time_raw"] > 0)
    df_rec = df_rec[valid_mask].copy()
    cleaning_log["after_remove_invalid_time"] = len(df_rec)

    # Event-specific physiological range filters
    time_ranges = {
        "100m": (inc["min_time_100m"], inc["max_time_100m"]),
        "200m": (inc["min_time_200m"], inc["max_time_200m"]),
        "60m":  (inc["min_time_60m"],  inc["max_time_60m"]),
        "400m": (17.0, 80.0),
    }
    masks = []
    for event, (lo, hi) in time_ranges.items():
        event_mask = df_rec["event"] == event
        range_mask = (df_rec["time_raw"] >= lo) & (df_rec["time_raw"] <= hi)
        masks.append(~event_mask | range_mask)
    combined = masks[0]
    for m in masks[1:]:
        combined = combined & m
    df_rec = df_rec[combined].copy()
    cleaning_log["after_physio_range"] = len(df_rec)

    # --- Step 1b: Remove 100 m hurdles records mis-filed under event == "100m" ---
    # Their event_full names a hurdles event (contains the character 栏, e.g.
    # "女子100米栏决赛") and their times, 11.08 to 19.98 s, all fall inside the
    # 100 m validity range, so no other filter removes them. They are a different
    # event and are excluded here, in response to Reviewer 4, round 3.
    hurdles = (
        (df_rec["event"] == "100m")
        & df_rec["event_full"].astype(str).str.contains("栏", na=False)
    )
    cleaning_log["hurdles_100m_removed"] = int(hurdles.sum())
    cleaning_log["hurdles_100m_athletes"] = int(
        df_rec.loc[hurdles, "athlete_id"].nunique()
    )
    df_rec = df_rec[~hurdles].copy()
    cleaning_log["after_remove_100m_hurdles"] = len(df_rec)

    # --- Step 2: Remove athletes without DOB ---
    athletes_with_dob = set(df_ath.loc[df_ath["dob"].notna(), "athlete_id"])
    df_rec = df_rec[df_rec["athlete_id"].isin(athletes_with_dob)].copy()
    cleaning_log["after_remove_no_dob"] = len(df_rec)

    # --- Step 3: Compute age_at_comp ---
    dob_map = df_ath.set_index("athlete_id")["dob"].to_dict()
    df_rec["dob"] = df_rec["athlete_id"].map(dob_map)
    df_rec["age_at_comp"] = (
        (df_rec["competition_date"] - df_rec["dob"]).dt.total_seconds() / (365.25 * 86400)
    )

    # --- Step 4: Remove records with unreasonable age ---
    age_mask = (df_rec["age_at_comp"] >= inc["min_age"]) & (df_rec["age_at_comp"] <= inc["max_age"])
    df_rec = df_rec[age_mask].copy()
    cleaning_log["after_age_filter"] = len(df_rec)

    # --- Step 5: Classify round_type ---
    df_rec["round_type"] = df_rec["event_full"].apply(classify_round)

    # --- Step 6: Flag indoor events ---
    df_rec["is_indoor"] = df_rec.apply(
        lambda r: is_indoor(r["event"], r.get("event_full", ""), r.get("competition_name", "")),
        axis=1,
    )

    # --- Step 7: Deduplicate ---
    n_before = len(df_rec)
    df_rec = df_rec.drop_duplicates(
        subset=["athlete_id", "event", "competition_date", "time_raw"],
        keep="first",
    )
    cleaning_log["dedup_removed"] = n_before - len(df_rec)
    cleaning_log["records_final"] = len(df_rec)

    # Drop temporary dob column from records (it's in athletes table)
    df_rec = df_rec.drop(columns=["dob"], errors="ignore")

    # --- Also filter athletes table to those with DOB ---
    df_ath_clean = df_ath[df_ath["dob"].notna()].copy()
    cleaning_log["athletes_with_dob"] = len(df_ath_clean)

    # Merge sex into records for downstream steps
    sex_map = df_ath.set_index("athlete_id")["sex"].to_dict()
    df_rec["sex"] = df_rec["athlete_id"].map(sex_map)

    # --- Step 8: Log summary ---
    logger.info("=" * 60)
    logger.info("CLEANING SUMMARY")
    logger.info("=" * 60)
    for k, v in cleaning_log.items():
        logger.info(f"  {k}: {v:,}")

    if len(df_rec) > 0:
        logger.info(f"  Events: {df_rec['event'].value_counts().to_dict()}")
        logger.info(f"  Round types: {df_rec['round_type'].value_counts().to_dict()}")
        logger.info(f"  Indoor: {df_rec['is_indoor'].sum():,}")

    write_parquet(df_rec, interim / "cleaned_records.parquet")
    write_parquet(df_ath_clean, interim / "cleaned_athletes.parquet")
    save_results_meta(cleaning_log, interim / "cleaning_log.json")


if __name__ == "__main__":
    main()
