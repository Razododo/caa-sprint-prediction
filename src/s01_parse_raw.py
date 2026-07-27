"""
Step 1: Parse raw CSV files from CAA database → unified parquet tables.

Input:  data/raw/*.csv (one CSV per athlete, format described in ENGINEERING_SPEC.md §2.1)
Output: data/interim/parsed_athletes.parquet
        data/interim/all_records_raw.parquet

Usage: python src/s01_parse_raw.py
"""
import logging
import re
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.io import load_config, ensure_dirs, write_parquet

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RESULT_PATTERN = re.compile(r'^([\d.]+)\s*(?:\(([+-]?[\d.]+)\))?$')

EVENT_MAP = {
    "60米": "60m",
    "100米": "100m",
    "200米": "200m",
    "400米": "400m",
    "800米": "800m",
    "1500米": "1500m",
    "3000米": "3000m",
    "5000米": "5000m",
    "10000米": "10000m",
}

INVALID_RESULTS = frozenset({"DNS", "DNF", "DQ", "w", "–", "-", "", "DSQ", "NM"})


def normalize_event(event_str: str) -> str:
    """
    Normalize Chinese event names to standardized codes.
    "男子组100米" → "100m", "男子U16组100米" → "100m", etc.
    """
    if not event_str:
        return "other"
    for cn, en in EVENT_MAP.items():
        if cn in event_str:
            return en
    return "other"


def parse_result(result_str: str) -> tuple[float | None, float | None]:
    """
    Parse result string into (time_seconds, wind_speed).

    Examples:
        "10.54 (+0.80)" → (10.54, 0.80)
        "09.92"          → (9.92, None)
        "DNS"            → (None, None)
    """
    if not result_str or result_str.strip() in INVALID_RESULTS:
        return None, None

    result_str = result_str.strip()
    match = RESULT_PATTERN.match(result_str)
    if match:
        time_val = float(match.group(1))
        wind_val = float(match.group(2)) if match.group(2) else None
        return time_val, wind_val

    try:
        return float(result_str), None
    except ValueError:
        return None, None


def parse_single_csv(filepath: Path) -> tuple[dict | None, list[dict]]:
    """
    Parse a single athlete CSV file.

    Returns:
        Tuple of (athlete_metadata_dict, list_of_record_dicts).
        Returns (None, []) if parsing fails.
    """
    try:
        text = filepath.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = filepath.read_text(encoding="gbk")
        except Exception:
            return None, []

    lines = text.strip().split("\n")
    if len(lines) < 8:
        return None, []

    # --- Parse header (lines 0–7, 0-indexed) ---
    def val(line: str) -> str:
        parts = line.split(",", 1)
        return parts[1].strip() if len(parts) > 1 else ""

    name = val(lines[1])
    dob_str = val(lines[2])
    team_raw = val(lines[3])
    sex_cn = val(lines[4])
    reg_no = val(lines[5])
    height_str = val(lines[6])
    weight_str = val(lines[7])

    sex = "M" if sex_cn == "男" else "F"

    dob = None
    if dob_str:
        try:
            dob = pd.Timestamp(dob_str)
        except Exception:
            pass

    height = None
    try:
        h = float(height_str)
        height = h if h > 0 else None
    except (ValueError, TypeError):
        pass

    weight = None
    try:
        w = float(weight_str)
        weight = w if w > 0 else None
    except (ValueError, TypeError):
        pass

    team_list = [t.strip() for t in team_raw.split("/") if t.strip()]

    athlete = {
        "athlete_id": reg_no if reg_no else f"_no_reg_{name}_{dob_str}_{sex}",
        "name": name,
        "dob": dob,
        "sex": sex,
        "height_cm": height,
        "weight_kg": weight,
        "team_history": team_raw,
        "team_list": team_list,
        "source_file": filepath.name,
    }

    # --- Parse records from "个人荣誉 - 历史最好成绩" section ---
    records: list[dict] = []
    in_records = False
    header_seen = False

    for line in lines[8:]:
        line = line.strip()

        if "个人最好成绩" in line:
            break

        if "个人荣誉" in line and "历史最好成绩" in line:
            in_records = True
            continue

        if in_records and not header_seen:
            if line.startswith("项目,"):
                header_seen = True
            continue

        if not header_seen:
            continue

        if not line:
            continue

        parts = line.split(",")
        if len(parts) < 6:
            continue

        event_raw = parts[0].strip()
        comp_name = parts[1].strip()
        event_full = parts[2].strip() if len(parts) > 2 else ""
        rank_str = parts[3].strip() if len(parts) > 3 else ""
        result_str = parts[4].strip() if len(parts) > 4 else ""
        date_str = parts[5].strip() if len(parts) > 5 else ""
        team_at = parts[6].strip() if len(parts) > 6 else ""
        reg_unit = parts[7].strip() if len(parts) > 7 else ""

        event_norm = normalize_event(event_raw)
        if event_norm == "other":
            continue

        time_val, wind_val = parse_result(result_str)
        if time_val is None:
            continue

        rank = None
        try:
            rank = int(rank_str)
        except (ValueError, TypeError):
            pass

        comp_date = None
        if date_str:
            try:
                comp_date = pd.Timestamp(date_str)
            except Exception:
                pass

        records.append({
            "athlete_id": athlete["athlete_id"],
            "event": event_norm,
            "competition_name": comp_name,
            "event_full": event_full,
            "rank": rank,
            "time_raw": time_val,
            "wind_speed": wind_val,
            "competition_date": comp_date,
            "team_at_comp": team_at,
            "registration_unit": reg_unit,
        })

    return athlete, records


def main() -> None:
    config = load_config()
    ensure_dirs(config)

    raw_dir = Path(config["paths"]["raw_data"])
    csv_files = sorted(raw_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {raw_dir}")

    athletes: list[dict] = []
    all_records: list[dict] = []
    parse_errors = 0

    for fp in tqdm(csv_files, desc="Parsing CSVs"):
        try:
            meta, records = parse_single_csv(fp)
            if meta is not None:
                athletes.append(meta)
                all_records.extend(records)
        except Exception as e:
            parse_errors += 1
            logger.warning(f"Failed to parse {fp.name}: {e}")

    logger.info(f"Parsed {len(athletes)} athletes, {len(all_records)} records, {parse_errors} errors")

    df_athletes = pd.DataFrame(athletes)
    df_records = pd.DataFrame(all_records) if all_records else pd.DataFrame()

    if len(df_athletes) > 0:
        df_athletes["dob"] = pd.to_datetime(df_athletes["dob"])
    if len(df_records) > 0:
        df_records["competition_date"] = pd.to_datetime(df_records["competition_date"])

    # Deduplication: same athlete + event + date + time → keep first
    n_before = len(df_records)
    if len(df_records) > 0:
        df_records = df_records.drop_duplicates(
            subset=["athlete_id", "event", "competition_date", "time_raw"],
            keep="first",
        )
    logger.info(f"Dedup: {n_before} → {len(df_records)} records ({n_before - len(df_records)} removed)")

    interim = Path(config["paths"]["interim"])
    write_parquet(df_athletes, interim / "parsed_athletes.parquet")
    write_parquet(df_records, interim / "all_records_raw.parquet")

    # Summary
    logger.info(f"Athletes: {len(df_athletes)}")
    logger.info(f"  Male: {(df_athletes['sex'] == 'M').sum()}, Female: {(df_athletes['sex'] == 'F').sum()}")
    logger.info(f"  With DOB: {df_athletes['dob'].notna().sum()}")
    logger.info(f"  With height: {df_athletes['height_cm'].notna().sum()}")
    logger.info(f"  With weight: {df_athletes['weight_kg'].notna().sum()}")
    logger.info(f"Records: {len(df_records)}")
    if len(df_records) > 0:
        logger.info(f"  Events: {df_records['event'].value_counts().to_dict()}")
        logger.info(f"  Date range: {df_records['competition_date'].min()} to {df_records['competition_date'].max()}")


if __name__ == "__main__":
    main()
