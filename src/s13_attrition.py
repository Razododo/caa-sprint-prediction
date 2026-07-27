"""
Step 13: Attrition analysis — talent pipeline, dropout curves, province heterogeneity.

Uses ALL athletes with any 100m record (no minimum record count).

Input:  data/interim/cleaned_records.parquet
        data/interim/cleaned_athletes.parquet  (for province, max_team_level)
Output: results/tables/attrition_curve.csv
        results/tables/transition_rates.csv
        results/tables/province_heterogeneity.csv
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import load_config, ensure_dirs, read_parquet

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def compute_attrition_curve(
    rec_100m: pd.DataFrame, sex_map: dict[str, str]
) -> pd.DataFrame:
    """
    Age-based attrition curve for ages 12-35.

    Active at age X = athletes whose first_age <= X <= last_age.
    Dropout at age X = athletes whose last_age == X.
    """
    logger.info("Computing age-based attrition curve ...")

    rec_100m = rec_100m.copy()
    rec_100m["sex"] = rec_100m["athlete_id"].map(sex_map)

    age_bounds = rec_100m.groupby("athlete_id").agg(
        first_age=("age_at_comp", "min"),
        last_age=("age_at_comp", "max"),
        sex=("sex", "first"),
    )
    # Floor ages to integer for binning
    age_bounds["first_age_int"] = age_bounds["first_age"].astype(int)
    age_bounds["last_age_int"] = age_bounds["last_age"].astype(int)

    rows = []
    for sex_code, sex_label in [("M", "Male"), ("F", "Female")]:
        sub = age_bounds[age_bounds["sex"] == sex_code]
        total_athletes = len(sub)

        for age in range(12, 36):
            active = ((sub["first_age_int"] <= age) & (sub["last_age_int"] >= age)).sum()
            dropout = (sub["last_age_int"] == age).sum()
            new_entry = (sub["first_age_int"] == age).sum()
            dropout_rate = dropout / active if active > 0 else 0.0

            rows.append({
                "sex": sex_label,
                "age": age,
                "active_athletes": int(active),
                "new_entries": int(new_entry),
                "dropouts": int(dropout),
                "dropout_rate": round(dropout_rate, 4),
                "total_athletes": total_athletes,
            })

    df = pd.DataFrame(rows)

    for sex in ["Male", "Female"]:
        s = df[df["sex"] == sex]
        peak = s.loc[s["active_athletes"].idxmax()]
        peak_drop = s.loc[s["dropout_rate"].idxmax()]
        logger.info(
            f"  {sex}: peak active at age {int(peak['age'])} "
            f"({int(peak['active_athletes']):,}), "
            f"peak dropout rate at age {int(peak_drop['age'])} "
            f"({peak_drop['dropout_rate']:.1%})"
        )

    return df


def compute_transition_rates(athletes: pd.DataFrame) -> pd.DataFrame:
    """
    Team level transition rates: school→municipal→provincial→national.

    max_team_level represents the highest level an athlete ever reached.
    """
    logger.info("Computing team level transition rates ...")

    level_order = ["school", "municipal", "provincial", "national"]
    level_rank = {lv: i for i, lv in enumerate(level_order)}

    rows = []
    for sex_code, sex_label in [("M", "Male"), ("F", "Female")]:
        sub = athletes[athletes["sex"] == sex_code]
        total = len(sub)

        level_counts = {}
        for lv in level_order:
            level_counts[lv] = (sub["max_team_level"] == lv).sum()
        level_counts["other"] = (sub["max_team_level"] == "other").sum()

        # "Reached" counts: athletes whose max_team_level is >= a given level
        reached = {}
        for lv in level_order:
            reached[lv] = sum(
                (sub["max_team_level"] == lv2).sum()
                for lv2 in level_order
                if level_rank.get(lv2, -1) >= level_rank[lv]
            )

        # Transition rates
        school_to_prov = reached["provincial"] / reached["school"] if reached["school"] > 0 else 0
        prov_to_nat = reached["national"] / reached["provincial"] if reached["provincial"] > 0 else 0
        school_to_nat = reached["national"] / reached["school"] if reached["school"] > 0 else 0

        rows.append({
            "sex": sex_label,
            "total": total,
            "n_school": level_counts["school"],
            "n_municipal": level_counts["municipal"],
            "n_provincial": level_counts["provincial"],
            "n_national": level_counts["national"],
            "n_other": level_counts["other"],
            "reached_school_plus": reached["school"],
            "reached_provincial_plus": reached["provincial"],
            "reached_national": reached["national"],
            "school_to_provincial_pct": round(school_to_prov * 100, 2),
            "provincial_to_national_pct": round(prov_to_nat * 100, 2),
            "school_to_national_pct": round(school_to_nat * 100, 2),
        })

        logger.info(f"  {sex_label}: total={total:,}")
        for lv in level_order:
            logger.info(f"    {lv}: {level_counts[lv]:,} (max level)")
        logger.info(f"    other: {level_counts['other']:,}")
        logger.info(
            f"    Transitions: school→prov={school_to_prov:.1%}, "
            f"prov→nat={prov_to_nat:.1%}, school→nat={school_to_nat:.1%}"
        )

    return pd.DataFrame(rows)


def compute_province_heterogeneity(
    rec_100m: pd.DataFrame,
    athletes: pd.DataFrame,
    min_athletes: int = 200,
) -> pd.DataFrame:
    """
    Province-level heterogeneity: career span, PB, dropout rate at 18, national count.
    """
    logger.info(f"Computing province heterogeneity (min {min_athletes} athletes) ...")

    sex_map = athletes.set_index("athlete_id")["sex"].to_dict()
    prov_map = athletes.set_index("athlete_id")["province"].to_dict()
    level_map = athletes.set_index("athlete_id")["max_team_level"].to_dict()

    rec = rec_100m.copy()
    rec["sex"] = rec["athlete_id"].map(sex_map)
    rec["province"] = rec["athlete_id"].map(prov_map)

    # Per-athlete stats
    ath_stats = rec.groupby("athlete_id").agg(
        first_age=("age_at_comp", "min"),
        last_age=("age_at_comp", "max"),
        pb=("time_raw", "min"),
        sex=("sex", "first"),
        province=("province", "first"),
    )
    ath_stats["career_span"] = ath_stats["last_age"] - ath_stats["first_age"]
    ath_stats["max_team_level"] = ath_stats.index.map(level_map)
    ath_stats["last_age_int"] = ath_stats["last_age"].astype(int)

    rows = []
    province_counts = ath_stats["province"].value_counts()
    qualifying_provs = province_counts[province_counts >= min_athletes].index

    for prov in qualifying_provs:
        sub = ath_stats[ath_stats["province"] == prov]

        first_int = sub["first_age"].astype(int)
        active_at_18 = ((first_int <= 18) & (sub["last_age_int"] >= 18)).sum()
        dropout_at_18 = (sub["last_age_int"] == 18).sum()
        dropout_rate_18 = dropout_at_18 / active_at_18 if active_at_18 > 0 else np.nan

        rows.append({
            "province": prov,
            "n_athletes": len(sub),
            "median_career_span_years": round(sub["career_span"].median(), 2),
            "mean_lifetime_pb": round(sub["pb"].mean(), 3),
            "active_at_age_18": int(active_at_18),
            "dropout_at_age_18": int(dropout_at_18),
            "dropout_rate_age_18": round(dropout_rate_18, 4) if not np.isnan(dropout_rate_18) else np.nan,
            "n_national_level": int((sub["max_team_level"] == "national").sum()),
            "pct_male": round((sub["sex"] == "M").mean() * 100, 1),
        })

    df = pd.DataFrame(rows).sort_values("n_athletes", ascending=False)

    logger.info(f"  Provinces qualifying (≥{min_athletes}): {len(df)}")
    for _, r in df.head(10).iterrows():
        logger.info(
            f"    {r['province']}: n={r['n_athletes']:,}  "
            f"career={r['median_career_span_years']}yr  "
            f"PB={r['mean_lifetime_pb']:.2f}  "
            f"dropout@18={r['dropout_rate_age_18']:.1%}  "
            f"national={r['n_national_level']}"
        )

    return df


def main() -> None:
    config = load_config()
    ensure_dirs(config)

    interim = Path(config["paths"]["interim"])
    tables_dir = Path(config["paths"]["tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    records = read_parquet(interim / "cleaned_records.parquet")
    athletes = read_parquet(interim / "cleaned_athletes.parquet")

    rec_100m = records[records["event"] == "100m"].copy()
    sex_map = athletes.set_index("athlete_id")["sex"].to_dict()

    logger.info(
        f"100m records: {len(rec_100m):,} from "
        f"{rec_100m['athlete_id'].nunique():,} athletes"
    )
    logger.info(f"Athletes with metadata: {len(athletes):,}")

    # === A) Attrition curve ===
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS A: Age-Based Attrition Curve")
    logger.info("=" * 70)
    df_attrition = compute_attrition_curve(rec_100m, sex_map)
    df_attrition.to_csv(tables_dir / "attrition_curve.csv", index=False)
    logger.info(f"Saved attrition_curve.csv ({len(df_attrition)} rows)")

    # === B) Team level transition rates ===
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS B: Team Level Transition Rates")
    logger.info("=" * 70)
    df_transitions = compute_transition_rates(athletes)
    df_transitions.to_csv(tables_dir / "transition_rates.csv", index=False)
    logger.info(f"Saved transition_rates.csv")

    # === C) Province heterogeneity ===
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS C: Province Heterogeneity")
    logger.info("=" * 70)
    df_prov = compute_province_heterogeneity(rec_100m, athletes)
    df_prov.to_csv(tables_dir / "province_heterogeneity.csv", index=False)
    logger.info(f"Saved province_heterogeneity.csv ({len(df_prov)} provinces)")

    logger.info("\nAll attrition analyses complete.")


if __name__ == "__main__":
    main()
