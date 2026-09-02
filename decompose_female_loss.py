# -*- coding: utf-8 -*-
"""
把"剔除跨栏后 Group A 少掉的人"拆成几类,供回复信引用。

用法(项目根目录):
    python decompose_female_loss.py

只读两份 cleaned_records:
    data_interim_prehurdles/cleaned_records.parquet   剔除前
    data/interim/cleaned_records.parquet              剔除后
性别取自 cleaned_athletes.parquet(只读 athlete_id 和 sex 两列)。

Group A 的判定完全照 src/s10_group_split.py:
    在 100 m 记录集内,cutoff 前(age <= cutoff)的 100 m 记录 >= 2 条,
    且最后一条 100 m 记录的年龄 > cutoff。
脚本先用剔除前的数据复现已发表的 A 人数,对不上就停,不往下算。
"""
import sys
from pathlib import Path

import pandas as pd

MIN_PRE = 2                      # s10 的 MIN_PRE_CUTOFF_RECORDS
CUTOFFS = [16, 17, 18]
PUBLISHED = {                    # 已发表的 Group A 人数,用来验证判定逻辑
    ("M", 16): 958, ("M", 17): 2889, ("M", 18): 2662,
    ("F", 16): 452, ("F", 17): 664, ("F", 18): 522,
}

root = Path(".")
old_p = root / "data_interim_prehurdles/cleaned_records.parquet"
new_p = root / "data/interim/cleaned_records.parquet"
ath_p = root / "data/interim/cleaned_athletes.parquet"
for p in (old_p, new_p, ath_p):
    if not p.exists():
        sys.exit(f"找不到 {p}")


def load_100m(path):
    r = pd.read_parquet(path, columns=["athlete_id", "event", "age_at_comp"])
    r = r[(r["event"] == "100m") & r["age_at_comp"].notna()]
    return r[["athlete_id", "age_at_comp"]]


def group_a(rec, cutoff):
    """照 s10 的规则返回 Group A 的 athlete_id 集合。"""
    pre = rec[rec["age_at_comp"] <= cutoff].groupby("athlete_id").size()
    last = rec.groupby("athlete_id")["age_at_comp"].max()
    eligible = pre[pre >= MIN_PRE].index
    return set(last.loc[last.index.isin(eligible) & (last > cutoff)].index)


old_rec, new_rec = load_100m(old_p), load_100m(new_p)
sex = (pd.read_parquet(ath_p, columns=["athlete_id", "sex"])
         .drop_duplicates("athlete_id").set_index("athlete_id")["sex"])

print(f"剔除前 100 m 记录 {len(old_rec):,},运动员 {old_rec.athlete_id.nunique():,}")
print(f"剔除后 100 m 记录 {len(new_rec):,},运动员 {new_rec.athlete_id.nunique():,}")
print()

old_ids = set(old_rec.athlete_id.unique())
new_ids = set(new_rec.athlete_id.unique())
gone_entirely = old_ids - new_ids
print(f"完全退出 100 m 样本(一条平跑记录都不剩)的运动员:{len(gone_entirely)} 人")
gs = sex.reindex(sorted(gone_entirely))
print(f"  其中 男 {(gs == 'M').sum()} 人,女 {(gs == 'F').sum()} 人")
print()

# ---- 先验证判定逻辑能复现已发表人数 ----
oldA = {c: group_a(old_rec, c) for c in CUTOFFS}
newA = {c: group_a(new_rec, c) for c in CUTOFFS}
bad = False
for c in CUTOFFS:
    for s in ("M", "F"):
        got = sum(1 for a in oldA[c] if sex.get(a) == s)
        want = PUBLISHED[(s, c)]
        flag = "" if got == want else "   <-- 对不上"
        if got != want:
            bad = True
        print(f"  校验 {s}{c}: 用剔除前数据算出 A = {got:,},已发表 {want:,}{flag}")
if bad:
    sys.exit("\n判定逻辑没能复现已发表人数,下面的拆解不可信,已停止。")
print("\n判定逻辑复现无误,继续拆解。\n")

# ---- 拆解 ----
old_pre = {c: old_rec[old_rec.age_at_comp <= c].groupby("athlete_id").size() for c in CUTOFFS}
new_pre = {c: new_rec[new_rec.age_at_comp <= c].groupby("athlete_id").size() for c in CUTOFFS}
new_last = new_rec.groupby("athlete_id")["age_at_comp"].max()

rows = []
for c in CUTOFFS:
    for s in ("M", "F"):
        o = {a for a in oldA[c] if sex.get(a) == s}
        n = {a for a in newA[c] if sex.get(a) == s}
        lost, gained = o - n, n - o
        cat = {"完全没有平跑记录": 0, "cutoff 前记录不足 2 条": 0,
               "cutoff 后没有记录了": 0, "其他": 0}
        for a in lost:
            if a not in new_ids:
                cat["完全没有平跑记录"] += 1
            elif new_pre[c].get(a, 0) < MIN_PRE:
                cat["cutoff 前记录不足 2 条"] += 1
            elif new_last.get(a, -1) <= c:
                cat["cutoff 后没有记录了"] += 1
            else:
                cat["其他"] += 1
        rows.append(dict(cutoff=c, sex=s, A_old=len(o), A_new=len(n),
                         lost=len(lost), pct=round(100 * len(lost) / len(o), 1),
                         gained=len(gained), **cat))

df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(df.to_string(index=False))
print()
if df["其他"].sum():
    print("注意:有落到\"其他\"的人,说明还有没识别出的机制,别直接把这张表写进信里。")
if df["gained"].sum():
    print("注意:有新进 Group A 的人,数字要在信里一并说明。")
print()
print("给回复信用的一句话(女子,cutoff 16 / 17 / 18):")
f = df[df.sex == "F"].set_index("cutoff")
for c in CUTOFFS:
    r = f.loc[c]
    print(f"  cutoff {c}: {r.A_old} -> {r.A_new},少 {r.lost} 人({r.pct}%);"
          f"其中 {r['完全没有平跑记录']} 人已无任何平跑记录,"
          f"{r['cutoff 前记录不足 2 条']} 人 cutoff 前不足 2 条,"
          f"{r['cutoff 后没有记录了']} 人 cutoff 后已无记录")
