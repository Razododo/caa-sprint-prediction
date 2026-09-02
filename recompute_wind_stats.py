# -*- coding: utf-8 -*-
"""
重算 Methods 段 36 里的四个风速统计(剔除跨栏后口径)。

用法(项目根目录):
    python recompute_wind_stats.py

正文段 36 现在写的是:
    "Wind speed was parsed from result fields (available for 91.8% of records)"
    "Among 100-m records with a recorded wind reading, 2.19% exceeded +2.0 m s⁻¹,
     and 2.83% of athletes have a lifetime best set under such conditions;
     a further 8.5% of lifetime-best races carry no wind reading at all."

这四个数在剔除跨栏后都要重算。脚本同时按"全部记录"和"仅 100 m"两种分母打印
解析率,因为已提交稿里那个 91.8% 用的是哪一种口径无法从文本判断,由你对照选。
"""
import sys
from pathlib import Path

import pandas as pd

p = Path("data/interim/cleaned_records.parquet")
if not p.exists():
    sys.exit(f"找不到 {p}")

rec = pd.read_parquet(p)
h = (rec["event"] == "100m") & rec["event_full"].astype(str).str.contains("栏", na=False)
if int(h.sum()):
    sys.exit(f"数据里还有 {int(h.sum()):,} 条跨栏,口径不对")

r100 = rec[rec["event"] == "100m"].copy()
print(f"自检:总记录 {len(rec):,},100 m {len(r100):,}(应为 129,351)\n")

has_wind_all = rec["wind_speed"].notna()
has_wind_100 = r100["wind_speed"].notna()
print("解析率(正文现写 91.8%)")
print(f"  按全部记录: {100 * has_wind_all.mean():.2f}%  ({int(has_wind_all.sum()):,} / {len(rec):,})")
print(f"  按 100 m  : {100 * has_wind_100.mean():.2f}%  ({int(has_wind_100.sum()):,} / {len(r100):,})")
print()

# 有风读数的 100 m 记录里,超过 +2.0 的比例
w = r100.loc[has_wind_100, "wind_speed"]
print(f"超过 +2.0 m/s 的比例(正文现写 2.19%): {100 * (w > 2.0).mean():.2f}%"
      f"  ({int((w > 2.0).sum()):,} / {len(w):,})")

# 每名运动员的终生最好成绩那一场
idx = r100.groupby("athlete_id")["time_raw"].idxmin().dropna()
best = r100.loc[idx]
n = len(best)
illegal = (best["wind_speed"] > 2.0).sum()
nowind = best["wind_speed"].isna().sum()
print(f"终生最好成绩在 wind > +2.0 下取得的运动员(正文现写 2.83%): "
      f"{100 * illegal / n:.2f}%  ({int(illegal):,} / {n:,})")
print(f"终生最好成绩那一场没有风读数(正文现写 8.5%): "
      f"{100 * nowind / n:.2f}%  ({int(nowind):,} / {n:,})")
print()
print("把这几行发我,我改进正文段 36。")
