# -*- coding: utf-8 -*-
"""
Recompute every row of Main Table 1 under the analysis definitions used in the
published article, and print the values in the order in which they appear there.

Usage (from the project root):
    python recompute_table1.py

Inputs:
    data/interim/cleaned_records.parquet     cleaned 100 m records
    data/interim/cleaned_athletes.parquet    athlete_id, sex, height_cm, weight_kg
    data/processed/group_labels.parquet      cutoff 18 Group A membership, has_anthro
    results/tables/attrition_curve.csv       active athlete counts and dropout rates
    results/tables/transition_rates.csv      team-level transition rates

Before computing anything the script asserts that no 100 m hurdles records remain
and that the record count matches the 129,351 reported in the article.

Requires the full processed dataset, which is not distributed with this repository.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(".")
rec_p = root / "data/interim/cleaned_records.parquet"
ath_p = root / "data/interim/cleaned_athletes.parquet"
gl_p = root / "data/processed/group_labels.parquet"
for p in (rec_p, ath_p, gl_p):
    if not p.exists():
        sys.exit(f"Not found: {p}")

rec = pd.read_parquet(rec_p)
rec = rec[rec["event"] == "100m"]
ath = pd.read_parquet(ath_p, columns=["athlete_id", "sex", "height_cm", "weight_kg"])
ath = ath.drop_duplicates("athlete_id")
gl = pd.read_parquet(gl_p)

sex = ath.set_index("athlete_id")["sex"]
rec = rec.assign(sex=rec["athlete_id"].map(sex))

hurdles_left = rec["event_full"].astype(str).str.contains("栏", na=False).sum()
print(f"Check: {int(hurdles_left)} hurdles records remaining (expected 0), "
      f"{len(rec):,} 100 m records (expected 129,351)")
if hurdles_left or len(rec) != 129351:
    sys.exit("Input does not match the published analysis set; stopping.")
print()

# Values as printed in the submitted manuscript, shown beside each recomputed
# value so that every row can be checked one by one.
OLD = {
    "reg_M": "51,211", "reg_F": "7,006",
    "ath_M": "48,852", "ath_F": "6,517",
    "recs_M": "109,828", "recs_F": "21,967",
    "med_M": "2 (1–3)", "med_F": "2 (1–4)",
    "mean_M": "2.25", "mean_F": "3.37",
    "age_M": "12–45", "age_F": "12–42",
    "pb_M": "11.829 ± 0.594", "pb_F": "13.479 ± 0.956",
    "rng_M": "9.83–19.94", "rng_F": "10.97–19.91",
    "peak_M": "17 (25,688)", "peak_F": "17 (2,551)",
    "d18_M": "83.9%", "d18_F": "62.3%",
    "pkd_M": "83.9% (18)", "pkd_F": "72.2% (17)",
    "trans_M": "0.5%", "trans_F": "2.5%",
    "anth_M": "14,965 (29.2%)", "anth_F": "2,157 (30.8%)",
    "h_M": "177.6 ± 5.0", "h_F": "167.0 ± 4.9",
    "w_M": "64.9 ± 6.1", "w_F": "53.4 ± 5.2",
}


def row(label, key_m, key_f, new_m, new_f):
    print(f"{label:42s} M {OLD[key_m]:>16s} -> {new_m:<16s}   F {OLD[key_f]:>16s} -> {new_f}")


# 1. Total registered athletes across all events, unaffected by the exclusion.
nreg = ath.groupby("sex").size()
row("Total registered athletes (all events)", "reg_M", "reg_F",
    f"{nreg.get('M', 0):,}", f"{nreg.get('F', 0):,}")

# 2. Athletes holding at least one 100 m record.
nath = rec.groupby("sex")["athlete_id"].nunique()
row("Athletes with >=1 100-m record", "ath_M", "ath_F",
    f"{nath.get('M', 0):,}", f"{nath.get('F', 0):,}")

# 3. Total 100 m records.
nrec = rec.groupby("sex").size()
row("Total 100-m records", "recs_M", "recs_F",
    f"{nrec.get('M', 0):,}", f"{nrec.get('F', 0):,}")

# 4, 5. Records per athlete, median (IQR) and mean.
per = rec.groupby(["sex", "athlete_id"]).size().rename("k").reset_index()
med, mean = {}, {}
for s in ("M", "F"):
    v = per.loc[per.sex == s, "k"]
    med[s] = f"{int(v.median())} ({int(v.quantile(.25))}–{int(v.quantile(.75))})"
    mean[s] = f"{v.mean():.2f}"
row("100-m records per athlete, median (IQR)", "med_M", "med_F", med["M"], med["F"])
row("100-m records per athlete, mean", "mean_M", "mean_F", mean["M"], mean["F"])
print(f"{'  (records per athlete, both sexes)':42s}     "
      f"{len(rec) / rec.athlete_id.nunique():.2f}")

# 6. Age range.
age = {}
for s in ("M", "F"):
    v = rec.loc[rec.sex == s, "age_at_comp"].dropna()
    age[s] = f"{int(np.floor(v.min()))}–{int(np.floor(v.max()))}"
row("Age range (years)", "age_M", "age_F", age["M"], age["F"])

# 7, 8. Lifetime personal best.
pb = rec.groupby("athlete_id")["time_raw"].min().rename("pb").reset_index()
pb["sex"] = pb["athlete_id"].map(sex)
pbm, pbr = {}, {}
for s in ("M", "F"):
    v = pb.loc[pb.sex == s, "pb"]
    pbm[s] = f"{v.mean():.3f} ± {v.std():.3f}"
    pbr[s] = f"{v.min():.2f}–{v.max():.2f}"
row("Lifetime PB, mean ± SD (s)", "pb_M", "pb_F", pbm["M"], pbm["F"])
row("Lifetime PB range (s)", "rng_M", "rng_F", pbr["M"], pbr["F"])

# 9, 10, 11. Attrition, taken from attrition_curve.csv, the source behind Fig 1.
ac = pd.read_csv("results/tables/attrition_curve.csv")
pk, d18, pkd = {}, {}, {}
for s, lab in (("M", "Male"), ("F", "Female")):
    a = ac[ac.sex == lab]
    i = a.active_athletes.idxmax()
    pk[s] = f"{int(a.loc[i, 'age'])} ({int(a.loc[i, 'active_athletes']):,})"
    r18 = a[a.age == 18]
    d18[s] = f"{100 * float(r18.dropout_rate.iloc[0]):.1f}%"
    q = a[a.active_athletes >= 100]          # same n>=100 threshold as Fig 1
    j = q.dropout_rate.idxmax()
    pkd[s] = f"{100 * float(q.loc[j, 'dropout_rate']):.1f}% ({int(q.loc[j, 'age'])})"
row("Peak active age (n)", "peak_M", "peak_F", pk["M"], pk["F"])
row("Dropout rate at age 18", "d18_M", "d18_F", d18["M"], d18["F"])
row("Peak dropout rate (age)", "pkd_M", "pkd_F", pkd["M"], pkd["F"])

# 12. Team-level transition rate.
tr = pd.read_csv("results/tables/transition_rates.csv").set_index("sex")
row("School -> national transition rate", "trans_M", "trans_F",
    f"{tr.loc['Male', 'school_to_national_pct']:.1f}%",
    f"{tr.loc['Female', 'school_to_national_pct']:.1f}%")

# 13. Athletes carrying height and weight, as a share of all registered athletes.
VH, VW = (140, 210), (35, 120)
ok = (ath.height_cm.between(*VH) & ath.weight_kg.between(*VW))
na = ath[ok].groupby("sex").size()
row("Athletes with anthropometric data", "anth_M", "anth_F",
    f"{na.get('M', 0):,} ({100 * na.get('M', 0) / nreg.get('M', 1):.1f}%)",
    f"{na.get('F', 0):,} ({100 * na.get('F', 0) / nreg.get('F', 1):.1f}%)")

# 14, 15. Height and weight among cutoff 18 Group A athletes with measurements.
a18 = gl[(gl["cutoff_18_group"] == "A")]
a18 = a18.merge(ath, on="athlete_id", how="left", suffixes=("", "_a"))
a18 = a18[a18.height_cm.between(*VH) & a18.weight_kg.between(*VW)]
hh, ww, nn = {}, {}, {}
for s in ("M", "F"):
    v = a18[a18.sex == s]
    nn[s] = len(v)
    hh[s] = f"{v.height_cm.mean():.1f} ± {v.height_cm.std():.1f}"
    ww[s] = f"{v.weight_kg.mean():.1f} ± {v.weight_kg.std():.1f}"
row("Height, mean ± SD (cm)*", "h_M", "h_F", hh["M"], hh["F"])
row("Weight, mean ± SD (kg)*", "w_M", "w_F", ww["M"], ww["F"])
print(f"{'  (table note: cutoff 18 Group A with measurements, n)':42s} "
      f"M 592 -> {nn['M']}   F 227 -> {nn['F']}")

print()
print(f"100 m records after the exclusion: {rec.shape[0]:,}")
print("Records per athlete, both sexes = %.2f" % (len(rec) / rec.athlete_id.nunique()))
