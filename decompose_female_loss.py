# -*- coding: utf-8 -*-
"""
Decompose the reduction in Group A membership produced by excluding 100 m hurdles
records, according to the reason each athlete ceases to qualify.

Usage (from the project root):
    python decompose_female_loss.py

Group A membership is determined exactly as in src/s10_group_split.py: at least
MIN_PRE_CUTOFF_RECORDS pre-cutoff 100 m records, and a last 100 m record after the
cutoff age. The script first reproduces the pre-exclusion Group A counts and stops
if they do not match, so that the decomposition is computed only once the
membership rule is known to be faithful.

Inputs:
    data_interim_prehurdles/cleaned_records.parquet   records before the exclusion
    data/interim/cleaned_records.parquet              records after the exclusion
    data/interim/cleaned_athletes.parquet             athlete_id and sex

Requires the full processed dataset and a pre-exclusion snapshot of the interim
data, neither of which is distributed with this repository.
"""
import sys
from pathlib import Path

import pandas as pd

MIN_PRE = 2                      # MIN_PRE_CUTOFF_RECORDS in s10
CUTOFFS = [16, 17, 18]
PUBLISHED = {                    # published Group A counts, used to verify the rule
    ("M", 16): 958, ("M", 17): 2889, ("M", 18): 2662,
    ("F", 16): 452, ("F", 17): 664, ("F", 18): 522,
}

LOST_NO_FLAT = "no flat 100 m record left"
LOST_TOO_FEW_PRE = "fewer than 2 pre-cutoff records"
LOST_NONE_AFTER = "no record after the cutoff"
LOST_OTHER = "other"

root = Path(".")
old_p = root / "data_interim_prehurdles/cleaned_records.parquet"
new_p = root / "data/interim/cleaned_records.parquet"
ath_p = root / "data/interim/cleaned_athletes.parquet"
for p in (old_p, new_p, ath_p):
    if not p.exists():
        sys.exit(f"Not found: {p}")


def load_100m(path):
    r = pd.read_parquet(path, columns=["athlete_id", "event", "age_at_comp"])
    r = r[(r["event"] == "100m") & r["age_at_comp"].notna()]
    return r[["athlete_id", "age_at_comp"]]


def group_a(rec, cutoff):
    """Return the set of Group A athlete_ids under the s10 membership rule."""
    pre = rec[rec["age_at_comp"] <= cutoff].groupby("athlete_id").size()
    last = rec.groupby("athlete_id")["age_at_comp"].max()
    eligible = pre[pre >= MIN_PRE].index
    return set(last.loc[last.index.isin(eligible) & (last > cutoff)].index)


old_rec, new_rec = load_100m(old_p), load_100m(new_p)
sex = (pd.read_parquet(ath_p, columns=["athlete_id", "sex"])
         .drop_duplicates("athlete_id").set_index("athlete_id")["sex"])

print(f"Before the exclusion: {len(old_rec):,} 100 m records, "
      f"{old_rec.athlete_id.nunique():,} athletes")
print(f"After the exclusion:  {len(new_rec):,} 100 m records, "
      f"{new_rec.athlete_id.nunique():,} athletes")
print()

old_ids = set(old_rec.athlete_id.unique())
new_ids = set(new_rec.athlete_id.unique())
gone_entirely = old_ids - new_ids
print(f"Athletes leaving the 100 m sample entirely (no flat record left): {len(gone_entirely)}")
gs = sex.reindex(sorted(gone_entirely))
print(f"  of whom {(gs == 'M').sum()} male, {(gs == 'F').sum()} female")
print()

# Verify that the membership rule reproduces the published counts.
oldA = {c: group_a(old_rec, c) for c in CUTOFFS}
newA = {c: group_a(new_rec, c) for c in CUTOFFS}
bad = False
for c in CUTOFFS:
    for s in ("M", "F"):
        got = sum(1 for a in oldA[c] if sex.get(a) == s)
        want = PUBLISHED[(s, c)]
        flag = "" if got == want else "   <-- mismatch"
        if got != want:
            bad = True
        print(f"  check {s}{c}: pre-exclusion data gives A = {got:,}, published {want:,}{flag}")
if bad:
    sys.exit("\nThe membership rule does not reproduce the published counts; stopping.")
print("\nMembership rule reproduces the published counts; continuing.\n")

# Decomposition.
old_pre = {c: old_rec[old_rec.age_at_comp <= c].groupby("athlete_id").size() for c in CUTOFFS}
new_pre = {c: new_rec[new_rec.age_at_comp <= c].groupby("athlete_id").size() for c in CUTOFFS}
new_last = new_rec.groupby("athlete_id")["age_at_comp"].max()

rows = []
for c in CUTOFFS:
    for s in ("M", "F"):
        o = {a for a in oldA[c] if sex.get(a) == s}
        n = {a for a in newA[c] if sex.get(a) == s}
        lost, gained = o - n, n - o
        cat = {LOST_NO_FLAT: 0, LOST_TOO_FEW_PRE: 0, LOST_NONE_AFTER: 0, LOST_OTHER: 0}
        for a in lost:
            if a not in new_ids:
                cat[LOST_NO_FLAT] += 1
            elif new_pre[c].get(a, 0) < MIN_PRE:
                cat[LOST_TOO_FEW_PRE] += 1
            elif new_last.get(a, -1) <= c:
                cat[LOST_NONE_AFTER] += 1
            else:
                cat[LOST_OTHER] += 1
        rows.append(dict(cutoff=c, sex=s, A_old=len(o), A_new=len(n),
                         lost=len(lost), pct=round(100 * len(lost) / len(o), 1),
                         gained=len(gained), **cat))

df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(df.to_string(index=False))
print()
if df[LOST_OTHER].sum():
    print("Note: some athletes fall into 'other', so a mechanism remains unaccounted for.")
if df["gained"].sum():
    print("Note: some athletes enter Group A after the exclusion; report those counts too.")
print()
print("Summary for the female cells (cutoff 16 / 17 / 18):")
f = df[df.sex == "F"].set_index("cutoff")
for c in CUTOFFS:
    r = f.loc[c]
    print(f"  cutoff {c}: {r.A_old} -> {r.A_new}, a loss of {r.lost} ({r.pct}%); "
          f"{r[LOST_NO_FLAT]} have no flat 100 m record left, "
          f"{r[LOST_TOO_FEW_PRE]} fall below two pre-cutoff records, "
          f"{r[LOST_NONE_AFTER]} have no record after the cutoff")
