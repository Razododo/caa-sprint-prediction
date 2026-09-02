# -*- coding: utf-8 -*-
"""
重算 Table 1 的每一行(剔除跨栏后口径),并打印可直接抄进表里的数字。

用法(项目根目录):
    python recompute_table1.py

读:
    data/interim/cleaned_records.parquet     剔除后的记录
    data/interim/cleaned_athletes.parquet    运动员表(只取 athlete_id / sex / height_cm / weight_kg)
    data/processed/group_labels.parquet      cutoff 18 Group A 与 has_anthro
    results/tables/attrition_curve.csv       活跃人数与流失率
    results/tables/transition_rates.csv      转项率

每一项都打印"旧值 -> 新值",旧值是已提交稿 Table 1 里的数,方便逐行核对。
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
        sys.exit(f"找不到 {p}")

rec = pd.read_parquet(rec_p)
rec = rec[rec["event"] == "100m"]
ath = pd.read_parquet(ath_p, columns=["athlete_id", "sex", "height_cm", "weight_kg"])
ath = ath.drop_duplicates("athlete_id")
gl = pd.read_parquet(gl_p)

sex = ath.set_index("athlete_id")["sex"]
rec = rec.assign(sex=rec["athlete_id"].map(sex))

hurdles_left = rec["event_full"].astype(str).str.contains("栏", na=False).sum()
print(f"自检:剩余跨栏记录 {int(hurdles_left)} 条(应为 0),100 m 记录 {len(rec):,} 条(应为 129,351)")
if hurdles_left or len(rec) != 129351:
    sys.exit("数据口径不对,已停止。")
print()

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
    print(f"{label:42s} 男 {OLD[key_m]:>16s} -> {new_m:<16s}   女 {OLD[key_f]:>16s} -> {new_f}")


# 1 注册总人数(全项目,不受本次剔除影响)
nreg = ath.groupby("sex").size()
row("Total registered athletes (all events)", "reg_M", "reg_F",
    f"{nreg.get('M', 0):,}", f"{nreg.get('F', 0):,}")

# 2 有 100 m 记录的人数
nath = rec.groupby("sex")["athlete_id"].nunique()
row("Athletes with >=1 100-m record", "ath_M", "ath_F",
    f"{nath.get('M', 0):,}", f"{nath.get('F', 0):,}")

# 3 100 m 记录数
nrec = rec.groupby("sex").size()
row("Total 100-m records", "recs_M", "recs_F",
    f"{nrec.get('M', 0):,}", f"{nrec.get('F', 0):,}")

# 4/5 每人记录数 中位数(IQR)与均值
per = rec.groupby(["sex", "athlete_id"]).size().rename("k").reset_index()
med, mean = {}, {}
for s in ("M", "F"):
    v = per.loc[per.sex == s, "k"]
    med[s] = f"{int(v.median())} ({int(v.quantile(.25))}–{int(v.quantile(.75))})"
    mean[s] = f"{v.mean():.2f}"
row("100-m records per athlete, median (IQR)", "med_M", "med_F", med["M"], med["F"])
row("100-m records per athlete, mean", "mean_M", "mean_F", mean["M"], mean["F"])
print(f"{'  (合计每人记录数均值)':42s}     {len(rec) / rec.athlete_id.nunique():.2f}   "
      f"(正文段 29 现写 2.38)")

# 6 年龄范围
age = {}
for s in ("M", "F"):
    v = rec.loc[rec.sex == s, "age_at_comp"].dropna()
    age[s] = f"{int(np.floor(v.min()))}–{int(np.floor(v.max()))}"
row("Age range (years)", "age_M", "age_F", age["M"], age["F"])

# 7/8 终生 PB
pb = rec.groupby("athlete_id")["time_raw"].min().rename("pb").reset_index()
pb["sex"] = pb["athlete_id"].map(sex)
pbm, pbr = {}, {}
for s in ("M", "F"):
    v = pb.loc[pb.sex == s, "pb"]
    pbm[s] = f"{v.mean():.3f} ± {v.std():.3f}"
    pbr[s] = f"{v.min():.2f}–{v.max():.2f}"
row("Lifetime PB, mean ± SD (s)", "pb_M", "pb_F", pbm["M"], pbm["F"])
row("Lifetime PB range (s)", "rng_M", "rng_F", pbr["M"], pbr["F"])

# 9/10/11 流失(取自 attrition_curve.csv,与 Fig 1 同源)
ac = pd.read_csv("results/tables/attrition_curve.csv")
pk, d18, pkd = {}, {}, {}
for s, lab in (("M", "Male"), ("F", "Female")):
    a = ac[ac.sex == lab]
    i = a.active_athletes.idxmax()
    pk[s] = f"{int(a.loc[i, 'age'])} ({int(a.loc[i, 'active_athletes']):,})"
    r18 = a[a.age == 18]
    d18[s] = f"{100 * float(r18.dropout_rate.iloc[0]):.1f}%"
    q = a[a.active_athletes >= 100]          # 与 Fig 1 的 n>=100 门槛一致
    j = q.dropout_rate.idxmax()
    pkd[s] = f"{100 * float(q.loc[j, 'dropout_rate']):.1f}% ({int(q.loc[j, 'age'])})"
row("Peak active age (n)", "peak_M", "peak_F", pk["M"], pk["F"])
row("Dropout rate at age 18", "d18_M", "d18_F", d18["M"], d18["F"])
row("Peak dropout rate (age)", "pkd_M", "pkd_F", pkd["M"], pkd["F"])

# 12 转项率
tr = pd.read_csv("results/tables/transition_rates.csv").set_index("sex")
row("School -> national transition rate", "trans_M", "trans_F",
    f"{tr.loc['Male', 'school_to_national_pct']:.1f}%",
    f"{tr.loc['Female', 'school_to_national_pct']:.1f}%")

# 13 有身高体重的人数(分母是全项目注册人数,与已提交稿一致)
VH, VW = (140, 210), (35, 120)
ok = (ath.height_cm.between(*VH) & ath.weight_kg.between(*VW))
na = ath[ok].groupby("sex").size()
row("Athletes with anthropometric data", "anth_M", "anth_F",
    f"{na.get('M', 0):,} ({100 * na.get('M', 0) / nreg.get('M', 1):.1f}%)",
    f"{na.get('F', 0):,} ({100 * na.get('F', 0) / nreg.get('F', 1):.1f}%)")

# 14/15 cutoff 18 Group A 且有身高体重者的身高体重
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
print(f"{'  (表注里的 Group A cutoff18 有测量者 n)':42s} 男 592 -> {nn['M']}   女 227 -> {nn['F']}")

print()
print("另外两个正文要用的数:")
print(f"  剔除的跨栏记录按性别:", rec.shape[0], "(剔除后)")
print("  段 29 的每人记录数均值(合计) = %.2f" % (len(rec) / rec.athlete_id.nunique()))
