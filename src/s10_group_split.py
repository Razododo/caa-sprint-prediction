"""
Step 10: Create Group A/B labels for every athlete at every cutoff age.

Group A = athletes with ≥1 100m record AFTER the cutoff age (true prediction)
Group B = athletes whose LAST 100m record is at or before the cutoff (trivial)
Excluded = athletes with < 2 pre-cutoff 100m records

Input:  data/interim/cleaned_records.parquet
        data/interim/parsed_athletes.parquet
        data/interim/cleaned_athletes.parquet  (for province mapping)
Output: data/processed/group_labels.parquet
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import load_config, ensure_dirs, read_parquet, write_parquet

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

CUTOFF_AGES = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
MIN_PRE_CUTOFF_RECORDS = 2

VALID_HEIGHT_RANGE = (140, 210)
VALID_WEIGHT_RANGE = (35, 120)


def main() -> None:
    config = load_config()
    ensure_dirs(config)

    interim = Path(config["paths"]["interim"])
    processed = Path(config["paths"]["processed"])
    processed.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    records = read_parquet(interim / "cleaned_records.parquet")
    parsed_ath = read_parquet(interim / "parsed_athletes.parquet")
    cleaned_ath = read_parquet(interim / "cleaned_athletes.parquet")

    # 100m records only
    rec_100m = records[records["event"] == "100m"].copy()
    logger.info(
        f"100m records: {len(rec_100m):,} from "
        f"{rec_100m['athlete_id'].nunique():,} athletes"
    )

    # All unique 100m athlete IDs
    all_100m_ids = set(rec_100m["athlete_id"].unique())

    # ------------------------------------------------------------------
    # Athlete-level aggregates
    # ------------------------------------------------------------------
    lifetime_pb = rec_100m.groupby("athlete_id")["time_raw"].min()
    lifetime_pb.name = "lifetime_pb"

    last_100m_age = rec_100m.groupby("athlete_id")["age_at_comp"].max()
    last_100m_age.name = "last_100m_age"

    # ------------------------------------------------------------------
    # Athlete metadata: sex, height, weight, province
    # ------------------------------------------------------------------
    # height_cm=0 and weight_kg=0 → NaN (safety; s01 already does this)
    parsed_ath = parsed_ath.copy()
    # The April re-parse of parsed_athletes.parquet introduced duplicate
    # athlete_id rows (72,244 rows for 72,224 unique ids). A duplicated index
    # makes ath_meta.at[aid, ...] return a Series rather than a scalar, which
    # the published run never hit because that run predates the re-parse.
    # Keep the first row per athlete so the index is unique.
    _n_before = len(parsed_ath)
    parsed_ath = parsed_ath.drop_duplicates(subset="athlete_id", keep="first")
    if len(parsed_ath) != _n_before:
        logger.warning(
            f"parsed_athletes: dropped {_n_before - len(parsed_ath):,} duplicate "
            f"athlete_id rows ({_n_before:,} -> {len(parsed_ath):,})"
        )
    parsed_ath.loc[parsed_ath["height_cm"] == 0, "height_cm"] = np.nan
    parsed_ath.loc[parsed_ath["weight_kg"] == 0, "weight_kg"] = np.nan

    ath_meta = parsed_ath[["athlete_id", "sex", "height_cm", "weight_kg"]].copy()

    # Province from cleaned_athletes (populated by s03)
    province_map = cleaned_ath.set_index("athlete_id")["province"].to_dict()
    ath_meta["province"] = ath_meta["athlete_id"].map(province_map)

    # has_anthro: valid height AND weight within physiological range
    valid_h = (
        ath_meta["height_cm"].notna()
        & (ath_meta["height_cm"] >= VALID_HEIGHT_RANGE[0])
        & (ath_meta["height_cm"] <= VALID_HEIGHT_RANGE[1])
    )
    valid_w = (
        ath_meta["weight_kg"].notna()
        & (ath_meta["weight_kg"] >= VALID_WEIGHT_RANGE[0])
        & (ath_meta["weight_kg"] <= VALID_WEIGHT_RANGE[1])
    )
    ath_meta["has_anthro"] = valid_h & valid_w

    # Restrict to athletes who appear in 100m records
    ath_meta = ath_meta[ath_meta["athlete_id"].isin(all_100m_ids)].copy()
    ath_meta = ath_meta.set_index("athlete_id")

    # Add lifetime_pb and last_100m_age
    ath_meta["lifetime_pb"] = lifetime_pb
    ath_meta["last_100m_age"] = last_100m_age

    logger.info(f"Athletes in output: {len(ath_meta):,}")

    # ------------------------------------------------------------------
    # Group labels per cutoff
    # ------------------------------------------------------------------
    # Pre-compute record counts per athlete per cutoff to avoid repeated filtering
    for cutoff in CUTOFF_AGES:
        pre = rec_100m[rec_100m["age_at_comp"] <= cutoff]
        pre_counts = pre.groupby("athlete_id").size()

        col = f"cutoff_{cutoff}_group"
        labels = pd.Series("excluded", index=ath_meta.index, name=col)

        eligible = pre_counts[pre_counts >= MIN_PRE_CUTOFF_RECORDS].index
        eligible_set = set(eligible) & set(ath_meta.index)

        for aid in eligible_set:
            if ath_meta.at[aid, "last_100m_age"] > cutoff:
                labels.at[aid] = "A"
            else:
                labels.at[aid] = "B"

        ath_meta[col] = labels

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 80)
    logger.info("GROUP SPLIT SUMMARY")
    logger.info("=" * 80)

    for cutoff in CUTOFF_AGES:
        col = f"cutoff_{cutoff}_group"
        counts = ath_meta[col].value_counts()
        n_a = counts.get("A", 0)
        n_b = counts.get("B", 0)
        n_ex = counts.get("excluded", 0)
        total = n_a + n_b + n_ex

        # Assertion: A + B + excluded = total athletes
        assert total == len(ath_meta), (
            f"Cutoff {cutoff}: {total} != {len(ath_meta)}"
        )

        # Per-sex breakdown
        for sex_code, sex_label in [("M", "Male"), ("F", "Female")]:
            sex_mask = ath_meta["sex"] == sex_code
            sex_counts = ath_meta.loc[sex_mask, col].value_counts()
            sa = sex_counts.get("A", 0)
            sb = sex_counts.get("B", 0)
            se = sex_counts.get("excluded", 0)
            logger.info(
                f"  Cutoff {cutoff:>2} {sex_label:>6}: "
                f"A={sa:>5,}  B={sb:>5,}  excl={se:>5,}  total={sa+sb+se:>5,}"
            )

            # Validation against diagnostic at cutoff 18
            if cutoff == 18 and sex_code == "M":
                if abs(sa - 2662) > 50:
                    logger.warning(
                        f"  ⚠ Male Group A at cutoff 18 = {sa}, "
                        f"expected ~2662 (diff={sa-2662})"
                    )
                else:
                    logger.info(f"  ✓ Male Group A at cutoff 18 = {sa} (matches diagnostic)")
            if cutoff == 18 and sex_code == "F":
                if abs(sa - 522) > 30:
                    logger.warning(
                        f"  ⚠ Female Group A at cutoff 18 = {sa}, "
                        f"expected ~522 (diff={sa-522})"
                    )
                else:
                    logger.info(f"  ✓ Female Group A at cutoff 18 = {sa} (matches diagnostic)")

    # Anthro subsample check at cutoff 18
    c18 = ath_meta[ath_meta["cutoff_18_group"] == "A"]
    for sex_code, sex_label, expected in [("M", "Male", 592), ("F", "Female", 227)]:
        n_anthro = c18[(c18["sex"] == sex_code) & c18["has_anthro"]].shape[0]
        logger.info(
            f"  Cutoff 18 Group A {sex_label} with anthro: {n_anthro} "
            f"(expected ~{expected})"
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out = ath_meta.reset_index()
    write_parquet(out, processed / "group_labels.parquet")
    logger.info(f"\nSaved group_labels.parquet: {len(out):,} rows × {out.shape[1]} cols")


if __name__ == "__main__":
    main()
