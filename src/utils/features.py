"""
Feature computation functions for the CAA Sprint pipeline.
Each function computes one group of features from pre-cutoff records.

CRITICAL: All functions receive ONLY pre-cutoff records. The caller
(s05_feature_engineer.py) is responsible for filtering by age cutoff.
"""
import numpy as np
import pandas as pd
from typing import Optional

from statsmodels.robust.robust_linear_model import RLM
import statsmodels.api as sm


def compute_trajectory_raw(records: pd.DataFrame) -> dict[str, float]:
    """
    Compute raw performance summary statistics from 100m records.

    Args:
        records: DataFrame with columns [time_raw, age_at_comp, ...]
                 Already filtered to pre-cutoff window.

    Returns:
        Dictionary of feature_name: value pairs.
    """
    times = records["time_raw"].dropna()
    if len(times) == 0:
        return {}

    return {
        "best_time_raw": times.min(),
        "mean_time_raw": times.mean(),
        "median_time_raw": times.median(),
        "sd_time_raw": times.std() if len(times) > 1 else 0.0,
        "cv_time_raw": (times.std() / times.mean()) if len(times) > 1 and times.mean() > 0 else 0.0,
        "worst_time_raw": times.max(),
        "range_time_raw": times.max() - times.min(),
        "q25_time_raw": times.quantile(0.25),
        "q75_time_raw": times.quantile(0.75),
    }


def compute_trajectory_wc(records: pd.DataFrame) -> dict[str, float]:
    """
    Compute wind-corrected performance summary statistics.
    Same metrics as raw but using time_wind_corrected column.
    Falls back to raw times for records without wind correction.
    """
    # Use wind-corrected where available, raw otherwise
    times = records["time_wind_corrected"].fillna(records["time_raw"]).dropna()
    if len(times) == 0:
        return {}

    return {
        "best_time_wc": times.min(),
        "mean_time_wc": times.mean(),
        "median_time_wc": times.median(),
        "sd_time_wc": times.std() if len(times) > 1 else 0.0,
        "cv_time_wc": (times.std() / times.mean()) if len(times) > 1 and times.mean() > 0 else 0.0,
        "worst_time_wc": times.max(),
        "range_time_wc": times.max() - times.min(),
        "q25_time_wc": times.quantile(0.25),
        "q75_time_wc": times.quantile(0.75),
    }


def compute_trajectory_dynamics(
    records: pd.DataFrame, min_records_for_slope: int = 3
) -> dict[str, float]:
    """
    Compute temporal trajectory and improvement metrics using Huber regression.

    Args:
        records: DataFrame with [time_raw, time_wind_corrected, age_at_comp]
        min_records_for_slope: Minimum records needed to fit slope.

    Returns:
        Dictionary of dynamics features.
    """
    result = {}
    times_raw = records["time_raw"].dropna()
    ages = records.loc[times_raw.index, "age_at_comp"]

    # Improvement slope (Huber robust regression: time ~ age)
    if len(times_raw) >= min_records_for_slope:
        try:
            X = sm.add_constant(ages.values)
            model = RLM(times_raw.values, X, M=sm.robust.norms.HuberT())
            fit = model.fit()
            result["improvement_slope_raw"] = fit.params[1]  # Negative = improving
            result["improvement_residual_sd"] = np.std(fit.resid)
        except Exception:
            result["improvement_slope_raw"] = np.nan
            result["improvement_residual_sd"] = np.nan
    else:
        result["improvement_slope_raw"] = np.nan
        result["improvement_residual_sd"] = np.nan

    # Wind-corrected slope
    times_wc = records["time_wind_corrected"].fillna(records["time_raw"]).dropna()
    ages_wc = records.loc[times_wc.index, "age_at_comp"]
    if len(times_wc) >= min_records_for_slope:
        try:
            X = sm.add_constant(ages_wc.values)
            model = RLM(times_wc.values, X, M=sm.robust.norms.HuberT())
            fit = model.fit()
            result["improvement_slope_wc"] = fit.params[1]
        except Exception:
            result["improvement_slope_wc"] = np.nan
    else:
        result["improvement_slope_wc"] = np.nan

    # First-to-best improvement
    if len(records) > 0:
        sorted_recs = records.sort_values("competition_date")
        first_time = sorted_recs["time_raw"].iloc[0]
        latest_time = sorted_recs["time_raw"].iloc[-1]
        best_time = times_raw.min()
        result["best_minus_first"] = first_time - best_time  # Positive = improved
        result["best_minus_latest"] = latest_time - best_time

        span = (sorted_recs["competition_date"].iloc[-1] - sorted_recs["competition_date"].iloc[0]).days / 365.25
        result["improvement_rate_annual"] = result["best_minus_first"] / span if span > 0 else 0.0
    else:
        result["best_minus_first"] = 0.0
        result["best_minus_latest"] = 0.0
        result["improvement_rate_annual"] = 0.0

    return result


def compute_career_structure(records: pd.DataFrame) -> dict[str, float]:
    """Compute career timeline and competition exposure features."""
    n = len(records)
    if n == 0:
        return {}

    dates = records["competition_date"]
    ages = records["age_at_comp"]
    span_days = (dates.max() - dates.min()).days
    span_years = span_days / 365.25

    return {
        "n_records": n,
        "n_competitions": dates.nunique(),
        "n_seasons": records["competition_date"].dt.year.nunique(),
        "career_span_years": span_years,
        "age_first_record": ages.min(),
        "age_best_record": ages.loc[records["time_raw"].idxmin()] if n > 0 else np.nan,
        "records_per_year": n / span_years if span_years > 0 else float(n),
    }


def compute_round_performance(records: pd.DataFrame) -> dict[str, float]:
    """
    Compute round-level features (NEW — CAA specific).
    Uses round_type column: 'heats', 'semi', 'final', 'unknown'.
    """
    finals = records[records["round_type"] == "final"]
    heats = records[records["round_type"] == "heats"]

    n_finals = len(finals)
    has_final = int(n_finals > 0)
    pct_finals = n_finals / len(records) if len(records) > 0 else 0.0

    # Heats-to-final time difference (positive = faster in finals)
    if len(finals) > 0 and len(heats) > 0:
        heats_final_diff = heats["time_raw"].mean() - finals["time_raw"].mean()
    else:
        heats_final_diff = 0.0

    return {
        "has_final": has_final,
        "n_finals": n_finals,
        "pct_finals": pct_finals,
        "heats_final_diff": heats_final_diff,
    }


def compute_wind_features(records: pd.DataFrame) -> dict[str, float]:
    """Compute wind exposure covariates."""
    wind = records["wind_speed"]
    n_total = len(records)
    n_missing = wind.isna().sum()
    wind_valid = wind.dropna()

    result = {
        "wind_missing_prop": n_missing / n_total if n_total > 0 else 1.0,
    }

    if len(wind_valid) > 0:
        result["wind_mean"] = wind_valid.mean()
        result["wind_sd"] = wind_valid.std() if len(wind_valid) > 1 else 0.0
        tail = wind_valid[wind_valid > 0]
        head = wind_valid[wind_valid < 0]
        result["wind_mean_tail"] = tail.mean() if len(tail) > 0 else 0.0
        result["wind_mean_head"] = head.mean() if len(head) > 0 else 0.0
    else:
        result["wind_mean"] = 0.0
        result["wind_sd"] = 0.0
        result["wind_mean_tail"] = 0.0
        result["wind_mean_head"] = 0.0

    return result


def compute_anthropometric(
    height_cm: Optional[float], weight_kg: Optional[float]
) -> dict[str, float]:
    """Compute anthropometric features from athlete metadata."""
    result = {
        "height_cm": height_cm if pd.notna(height_cm) else np.nan,
        "weight_kg": weight_kg if pd.notna(weight_kg) else np.nan,
        "height_missing": int(pd.isna(height_cm)),
        "weight_missing": int(pd.isna(weight_kg)),
    }

    if pd.notna(height_cm) and pd.notna(weight_kg) and height_cm > 0:
        result["bmi"] = weight_kg / (height_cm / 100) ** 2
    else:
        result["bmi"] = np.nan

    return result


def compute_era_features(records: pd.DataFrame, dob: pd.Timestamp) -> dict[str, float]:
    """Compute temporal context features."""
    years = records["competition_date"].dt.year
    if len(years) == 0:
        return {}

    return {
        "year_first": int(years.min()),
        "year_last": int(years.max()),
        "year_mean": years.mean(),
        "career_span_calendar": int(years.max() - years.min()),
        "decade_first": int(years.min() // 10 * 10),
        "decade_mode": int(years.mode().iloc[0] // 10 * 10) if len(years.mode()) > 0 else int(years.min() // 10 * 10),
        "birth_cohort_5yr": int(dob.year // 5 * 5) if pd.notna(dob) else np.nan,
    }
