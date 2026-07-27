"""
Step 4: Wind speed correction.

Input:  data/interim/cleaned_records.parquet
Output: Updated time_wind_corrected and wind_missing columns in cleaned_records.parquet

Formula (replicate WA paper exactly):
    t_adj = t_raw - wind × β
    β_male = 0.05, β_female = 0.06

- Records with wind data: apply correction
- Records without wind (indoor/missing): retain raw time, flag wind_missing=True
"""
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from utils.io import load_config, ensure_dirs, read_parquet, write_parquet

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def apply_wind_correction(
    time_raw: float, wind: float, beta: float
) -> float:
    """Apply linear wind correction: t_adj = t_raw - wind * beta."""
    if pd.isna(wind) or pd.isna(time_raw):
        return np.nan
    return time_raw - wind * beta


def main() -> None:
    config = load_config()
    ensure_dirs(config)

    interim = Path(config["paths"]["interim"])
    df = read_parquet(interim / "cleaned_records.parquet")

    beta_m = config["wind"]["beta_male"]
    beta_f = config["wind"]["beta_female"]

    df["wind_missing"] = df["wind_speed"].isna()

    beta = np.where(df["sex"] == "M", beta_m, beta_f)
    has_wind = df["wind_speed"].notna() & df["time_raw"].notna()

    df["time_wind_corrected"] = np.where(
        has_wind,
        df["time_raw"] - df["wind_speed"] * beta,
        np.nan,
    )

    n_corrected = has_wind.sum()
    n_missing = (~has_wind).sum()
    logger.info(f"Wind correction applied: {n_corrected:,} records corrected, {n_missing:,} missing wind")
    logger.info(f"  β_male={beta_m}, β_female={beta_f}")

    for ev in ["100m", "200m"]:
        ev_mask = df["event"] == ev
        ev_wind = has_wind & ev_mask
        logger.info(
            f"  {ev}: {ev_wind.sum():,}/{ev_mask.sum():,} with wind "
            f"({ev_wind.sum()/ev_mask.sum()*100:.1f}%)"
        )

    write_parquet(df, interim / "cleaned_records.parquet")


if __name__ == "__main__":
    main()
