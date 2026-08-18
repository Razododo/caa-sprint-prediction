"""
Reviewer 4, major comment #5 -- attrition recomputed on cohorts with adequate follow-up.
=========================================================================================
The published attrition curve (s13_attrition.py) defines

    active  at age X  =  first_age <= X <= last_age
    dropout at age X  =  last_age == X

An athlete whose final record simply falls near the end of the database is
therefore counted as a dropout even though no absence has actually been
observed. This conflates true dropout with administrative right-censoring, and
the effect is largest exactly where the paper's headline sits (age 18).

This script recomputes the curve requiring K years of potential observation
after age X: an athlete contributes to age X only if they were already at least
X + K years old at the database cut-off, so a genuine K-year absence could have
been seen.

K = 0 reproduces the published curve.

Output: results/tables/r4_attrition_by_followup.csv
"""
import sys
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent   # repo root; this script lives in r4_reanalysis/
sys.path.insert(0, str(HERE / "src"))

from utils.io import load_config, read_parquet      # noqa: E402

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


K_VALUES = [0, 1, 2, 3]
AGES = range(12, 36)


def main() -> None:
    require_full_dataset()
    config = load_config()
    interim = Path(config["paths"]["interim"])

    records = read_parquet(interim / "cleaned_records.parquet")
    athletes = read_parquet(interim / "cleaned_athletes.parquet")

    rec = records[records["event"] == "100m"].copy()
    rec["competition_date"] = pd.to_datetime(rec["competition_date"])
    data_end = rec["competition_date"].max()
    print(f"100 m database cut-off: {data_end.date()}")

    dob = pd.to_datetime(athletes.set_index("athlete_id")["dob"])
    sex_map = athletes.set_index("athlete_id")["sex"]

    bounds = rec.groupby("athlete_id").agg(
        first_age=("age_at_comp", "min"),
        last_age=("age_at_comp", "max"),
    )
    bounds["sex"] = bounds.index.map(sex_map)
    bounds["dob"] = bounds.index.map(dob)
    bounds = bounds.dropna(subset=["dob", "sex"])
    bounds["age_at_end"] = (data_end - bounds["dob"]).dt.days / 365.25
    bounds["first_age_int"] = bounds["first_age"].astype(int)
    bounds["last_age_int"] = bounds["last_age"].astype(int)

    print(f"athletes with DOB and >=1 100 m record: {len(bounds):,}")

    rows = []
    for K in K_VALUES:
        for sex_code, sex_label in [("M", "Male"), ("F", "Female")]:
            sub = bounds[bounds["sex"] == sex_code]
            for age in AGES:
                obs = sub[sub["age_at_end"] >= age + K]
                active = ((obs["first_age_int"] <= age) & (obs["last_age_int"] >= age)).sum()
                drop = (obs["last_age_int"] == age).sum()
                rows.append({
                    "followup_years_required": K,
                    "sex": sex_label,
                    "age": age,
                    "active_athletes": int(active),
                    "dropouts": int(drop),
                    "dropout_rate": round(drop / active, 4) if active else np.nan,
                })

    out = pd.DataFrame(rows)
    dest = HERE / "results/tables/r4_attrition_by_followup.csv"
    out.to_csv(dest, index=False)
    print(f"\nSaved {dest}")

    # --- headline comparison -------------------------------------------------
    print("\nDropout rate at the ages the manuscript reports:")
    print(f"{'sex':<8}{'age':>4}" + "".join(f"{'K='+str(k):>12}" for k in K_VALUES)
          + f"{'n active K=0':>14}{'n active K=2':>14}")
    print("-" * 78)
    for sex, age in [("Male", 18), ("Male", 17), ("Female", 17), ("Female", 18)]:
        line = f"{sex:<8}{age:>4}"
        for k in K_VALUES:
            v = out[(out.sex == sex) & (out.age == age) & (out.followup_years_required == k)]
            line += f"{v.dropout_rate.iloc[0]:>12.3f}"
        n0 = out[(out.sex == sex) & (out.age == age) & (out.followup_years_required == 0)].active_athletes.iloc[0]
        n2 = out[(out.sex == sex) & (out.age == age) & (out.followup_years_required == 2)].active_athletes.iloc[0]
        print(line + f"{n0:>14,}{n2:>14,}")

    print("\nPeak dropout age and rate by K:")
    for K in K_VALUES:
        for sex in ["Male", "Female"]:
            s = out[(out.followup_years_required == K) & (out.sex == sex)
                    & (out.active_athletes >= 50)]
            pk = s.loc[s.dropout_rate.idxmax()]
            print(f"  K={K} {sex:<7} peak at age {int(pk.age)}: {pk.dropout_rate:.1%} "
                  f"(n active {int(pk.active_athletes):,})")


if __name__ == "__main__":
    main()
