"""
Build the de-identified sample dataset released with the public repository.

Reads the internal modelling tables (data/processed/) and writes a small,
de-identified extract to data/sample/ that is safe to publish.

De-identification:
  - athlete_id is replaced by a salted SHA-256 digest. The salt is read from
    the CAA_DEID_SALT environment variable and is never committed, so the
    mapping cannot be reversed from the public files.
  - Fallback athlete_ids of the form "_no_reg_<name>_<dob>_<sex>" embed the
    athlete's name and full date of birth; hashing is what removes them.
  - No name, date of birth or team/club string is carried over.
  - province is replaced by an arbitrary group code. The province-disjoint
    cross-validation in s11 only needs a grouping variable, not the province
    name, and the name is a strong quasi-identifier.
  - Ages are released in whole years. In the internal tables an age in years is
    an exact whole number of days divided by 365.25, so a single age at full
    precision plus the competition date, which is public, reconstructs the date
    of birth exactly. last_100m_age is dropped outright: nothing in this
    repository reads it from the released file, and the cutoff_<age>_group
    columns already carry what the demo needs.
  - Every other float is rounded. Full double precision on a derived statistic
    such as mean_time_raw or cv_time_raw is a per-athlete fingerprint, and it
    also leaves inversion channels open: n_records / records_per_year recovers
    the career span in exact days even after career_span_years is rounded.

The audit at the end of this script is not decorative. It fails the build if any
released column still carries CJK text, a full date, or an age at day precision.

Usage:
    CAA_DEID_SALT="<your-secret-salt>" python scripts/make_public_sample.py
    CAA_DEID_SALT="..." python scripts/make_public_sample.py --n-group-a 1000
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
SAMPLE = PROJECT_ROOT / "data" / "sample"

SAMPLE_CUTOFFS = [16, 18, 20, 22]

# Columns that must never reach the public extract, whatever the source table.
FORBIDDEN_COLS = {"name", "athlete_name", "dob", "birth_date", "team", "club", "unit"}

# Dropped because they carry an age to the day and no released script reads them.
DROP_COLS = {"last_100m_age"}

# Ages in years. Released rounded to whole years.
AGE_YEAR_COLS = {"age_first_record", "age_best_record", "career_span_years"}

# All remaining floats are rounded to this many decimals.
FLOAT_DECIMALS = 4

# Grouping columns whose labels are replaced by arbitrary codes.
RECODE_COLS = {"province": "P"}

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

DAYS_PER_YEAR = 365.25
DAY_TOL = 1e-6


def hash_id(raw_id: str, salt: str) -> str:
    return hashlib.sha256((salt + str(raw_id)).encode("utf-8")).hexdigest()[:16]


def build_code_map(values: pd.Series, prefix: str, seed: int) -> dict:
    """Arbitrary, reproducible label -> code map.

    Codes are assigned by a seeded permutation so that they carry no ordering
    information: P00 is neither the alphabetically first province nor the
    largest one.
    """
    levels = sorted(values.dropna().astype(str).unique())
    order = np.random.default_rng(seed).permutation(len(levels))
    return {lvl: f"{prefix}{int(pos):02d}" for lvl, pos in zip(levels, order)}


def coarsen(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Apply the categorical and numeric coarsening rules."""
    out = df.copy()

    for col, prefix in RECODE_COLS.items():
        if col in out.columns:
            code_map = build_code_map(out[col], prefix, seed)
            out[col] = out[col].map(
                lambda v: code_map.get(str(v)) if pd.notna(v) else v
            )

    for col in out.columns:
        if out[col].dtype.kind != "f":
            continue
        if col in AGE_YEAR_COLS:
            out[col] = out[col].round(0)
        else:
            out[col] = out[col].round(FLOAT_DECIMALS)

    return out


def scrub(df: pd.DataFrame, salt: str, seed: int) -> pd.DataFrame:
    drop = [c for c in df.columns if c.lower() in FORBIDDEN_COLS or c in DROP_COLS]
    out = df.drop(columns=drop)
    out["athlete_id"] = out["athlete_id"].map(lambda x: hash_id(x, salt))
    return coarsen(out, seed)


def audit(df: pd.DataFrame, label: str) -> list:
    """Fail the build on residual identifying content.

    Three checks: forbidden or dropped columns that reappeared, free text that
    looks like a name or a full date, and any numeric column that still encodes
    an age at day precision.
    """
    problems = []

    for col in df.columns:
        if col.lower() in FORBIDDEN_COLS or col in DROP_COLS:
            problems.append(f"{label}.{col}: column should not be released")

    for col in df.columns:
        if df[col].dtype != object:
            continue
        vals = df[col].dropna().astype(str)
        if vals.str.contains(CJK_RE, regex=True).any():
            problems.append(f"{label}.{col}: contains CJK text")
        if vals.str.contains(DATE_RE, regex=True).any():
            problems.append(f"{label}.{col}: contains a full date")

    for col in df.columns:
        if df[col].dtype.kind != "f":
            continue
        v = df[col].dropna().to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        # Whole numbers are excluded: a count, or a whole-year age divisible by
        # four, is a whole number of days by arithmetic rather than by leakage.
        v = v[np.abs(v - np.round(v)) > DAY_TOL]
        if len(v) < 20:
            continue
        d = v * DAYS_PER_YEAR
        frac = float(np.mean(np.abs(d - np.round(d)) < DAY_TOL))
        if frac > 0.5:
            problems.append(
                f"{label}.{col}: {frac:.0%} of values are a whole number of days, "
                "so this column encodes an age at day precision"
            )

    return problems


def build_manifest(args) -> dict:
    return {
        "n_athletes_released": None,
        "sampling": (
            "Purposive demo sample, NOT a random sample of the cohort: Group A "
            "athletes are over-sampled so the primary analysis can be executed "
            "end to end. Do not use for inference about the CAA population."
        ),
        "n_group_a_per_sex": args.n_group_a,
        "n_background": args.n_background,
        "sampling_seed": args.seed,
        "id_scheme": "sha256(secret_salt + athlete_id)[:16]",
        "deidentification": {
            "dropped_columns": sorted(FORBIDDEN_COLS | DROP_COLS),
            "recoded_columns": {
                k: "arbitrary group code, seeded permutation, mapping not released"
                for k in RECODE_COLS
            },
            "ages": "rounded to whole years, so a date of birth cannot be reconstructed",
            "other_floats": f"rounded to {FLOAT_DECIMALS} decimals",
        },
        "condition_of_use": (
            "No attempt may be made to re-identify any individual in this extract, "
            "or to link it to any other dataset for that purpose."
        ),
        "files": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-group-a", type=int, default=750,
        help="Group A athletes to release per sex (Group A is the paper's primary "
             "analysis set and is rare, so it is deliberately over-sampled).",
    )
    parser.add_argument(
        "--n-background", type=int, default=500,
        help="Additional athletes drawn from the remainder (Group B / excluded).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    salt = os.environ.get("CAA_DEID_SALT")
    if not salt:
        sys.exit(
            "CAA_DEID_SALT is not set. Choose a secret salt, keep it out of the "
            "repository, and re-run:\n"
            '  CAA_DEID_SALT="<your-secret-salt>" python scripts/make_public_sample.py'
        )

    labels = pd.read_parquet(PROCESSED / "group_labels.parquet")
    labels = labels[labels["athlete_id"].notna()]

    rng = np.random.default_rng(args.seed)
    group_cols = [f"cutoff_{c}_group" for c in SAMPLE_CUTOFFS if f"cutoff_{c}_group" in labels]
    is_a = labels[group_cols].eq("A").any(axis=1)

    chosen = set()
    for sex in ("M", "F"):
        pool = labels.loc[is_a & (labels["sex"] == sex), "athlete_id"].unique()
        take = min(args.n_group_a, len(pool))
        chosen |= set(rng.choice(pool, size=take, replace=False))

    rest = labels.loc[~is_a, "athlete_id"].unique()
    take = min(args.n_background, len(rest))
    chosen |= set(rng.choice(rest, size=take, replace=False))

    SAMPLE.mkdir(parents=True, exist_ok=True)
    problems = []
    manifest = build_manifest(args)
    manifest["n_athletes_released"] = len(chosen)

    labels_out = scrub(labels[labels["athlete_id"].isin(chosen)].copy(), salt, args.seed)
    problems += audit(labels_out, "group_labels_sample")
    labels_out.to_csv(SAMPLE / "group_labels_sample.csv", index=False)
    manifest["files"]["group_labels_sample.csv"] = list(labels_out.shape)

    for cutoff in SAMPLE_CUTOFFS:
        src = PROCESSED / f"features_cutoff_{cutoff}.parquet"
        if not src.exists():
            continue
        feats = pd.read_parquet(src)
        sub = scrub(feats[feats["athlete_id"].isin(chosen)].copy(), salt, args.seed)
        problems += audit(sub, f"features_cutoff_{cutoff}_sample")
        fname = f"features_cutoff_{cutoff}_sample.csv"
        sub.to_csv(SAMPLE / fname, index=False)
        manifest["files"][fname] = list(sub.shape)
        print(f"{fname}: {sub.shape[0]} athletes x {sub.shape[1]} columns")

    if problems:
        print("\nDE-IDENTIFICATION AUDIT FAILED:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    (SAMPLE / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nDe-identification audit passed. Wrote {len(manifest['files'])} files to {SAMPLE}")


if __name__ == "__main__":
    main()
