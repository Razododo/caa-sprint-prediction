# -*- coding: utf-8 -*-
"""
Recompute the four wind-speed statistics reported in the Methods.

Usage (from the project root):
    python recompute_wind_stats.py

The script reports the wind-parsing rate against two denominators, all records and
100 m records only, and then, among 100 m records carrying a wind reading, the
proportion above +2.0 m s-1, the proportion of athletes whose lifetime best was set
above that threshold, and the proportion of lifetime-best races with no wind
reading at all.

Before computing anything it asserts that no 100 m hurdles records remain.

Requires the full processed dataset, which is not distributed with this repository.
"""
import sys
from pathlib import Path

import pandas as pd

p = Path("data/interim/cleaned_records.parquet")
if not p.exists():
    sys.exit(f"Not found: {p}")

rec = pd.read_parquet(p)
h = (rec["event"] == "100m") & rec["event_full"].astype(str).str.contains("栏", na=False)
if int(h.sum()):
    sys.exit(f"{int(h.sum()):,} hurdles records remain; input does not match the analysis set")

r100 = rec[rec["event"] == "100m"].copy()
print(f"Check: {len(rec):,} records in total, {len(r100):,} 100 m (expected 129,351)\n")

has_wind_all = rec["wind_speed"].notna()
has_wind_100 = r100["wind_speed"].notna()
print("Wind-parsing rate")
print(f"  all records: {100 * has_wind_all.mean():.2f}%  ({int(has_wind_all.sum()):,} / {len(rec):,})")
print(f"  100 m only : {100 * has_wind_100.mean():.2f}%  ({int(has_wind_100.sum()):,} / {len(r100):,})")
print()

# Share of 100 m records with a wind reading that exceed +2.0 m/s.
w = r100.loc[has_wind_100, "wind_speed"]
print(f"Above +2.0 m/s: {100 * (w > 2.0).mean():.2f}%"
      f"  ({int((w > 2.0).sum()):,} / {len(w):,})")

# The race in which each athlete set their lifetime best.
idx = r100.groupby("athlete_id")["time_raw"].idxmin().dropna()
best = r100.loc[idx]
n = len(best)
illegal = (best["wind_speed"] > 2.0).sum()
nowind = best["wind_speed"].isna().sum()
print(f"Athletes whose lifetime best was set with wind > +2.0 m/s: "
      f"{100 * illegal / n:.2f}%  ({int(illegal):,} / {n:,})")
print(f"Lifetime-best races with no wind reading: "
      f"{100 * nowind / n:.2f}%  ({int(nowind):,} / {n:,})")
