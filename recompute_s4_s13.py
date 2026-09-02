# -*- coding: utf-8 -*-
"""
补充表 S4(三种 boosting 框架对比)与 S13(省份组)重算。

用法(项目根目录):
    python recompute_s4_s13.py

S13 不需要额外依赖,先跑;S4 需要 xgboost 和 lightgbm,缺了会跳过并给出安装命令,
不会中断 S13 的输出。

S4 说明:原表用的重复次数与主表不同(GB 那列是 0.609 / 0.799 / 0.647,主表是
0.606 / 0.799 / 0.644),而且 xgboost 与 lightgbm 不在 requirements 里,原脚本
也没留在仓库中。这里统一改成与主表相同的 500 次重复、相同的 KFold 种子、相同的
中位数插补管道,所以重算后 S4 的 GB 那列应当与主表逐位相同 —— 这就成了这张表
自带的复现检验。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from utils.io import load_config, read_parquet          # noqa: E402
from utils.validation import random_cv                  # noqa: E402
import s11_controlled_analysis as s11                    # noqa: E402

from sklearn.pipeline import Pipeline                    # noqa: E402
from sklearn.impute import SimpleImputer                 # noqa: E402

VARIANT = "traj_wind"
S4_CELLS = [(16, "M", "Male"), (18, "M", "Male"), (18, "F", "Female")]
ALL_CELLS = [(c, s, lab) for c in (16, 17, 18) for s, lab in (("M", "Male"), ("F", "Female"))]

config = load_config()
interim = Path(config["paths"]["interim"])
processed = Path(config["paths"]["processed"])
records = read_parquet(interim / "cleaned_records.parquet")
athletes = read_parquet(interim / "cleaned_athletes.parquet")
gl = read_parquet(processed / "group_labels.parquet")
rec_100m = records[records["event"] == "100m"].copy()

n_folds = config["modeling"]["cv"]["n_folds"]
seed = config["modeling"]["cv"]["random_seed"]
n_rep = config["modeling"]["cv"]["n_repeats"]
n_jobs = config["modeling"]["cv"].get("n_jobs", 1)
registry = s11.load_feature_registry()

print(f"记录 {len(records):,} | 100 m {len(rec_100m):,} | group_labels {len(gl):,}")
print(f"CV: {n_folds} 折 × {n_rep} 次重复, seed={seed}, n_jobs={n_jobs}\n")

# ======================================================================
# S13:省份组
# ======================================================================
print("=" * 78)
print("Supplementary Table S13")
print("=" * 78)
prov = athletes.drop_duplicates("athlete_id").set_index("athlete_id")["province"]
rows13 = []
for cutoff, sx, lab in ALL_CELLS:
    ids = gl[(gl[f"cutoff_{cutoff}_group"] == "A") & (gl["sex"] == sx)]["athlete_id"]
    p = prov.reindex(ids).dropna()
    counts = p.value_counts()
    kept = counts[counts >= 5]
    rows13.append(dict(cell=f"{lab} {cutoff}", n_group_a=len(ids),
                       provinces_represented=int((counts > 0).sum()),
                       province_groups_used=int(len(kept)),
                       n_in_validation=int(kept.sum())))
s13 = pd.DataFrame(rows13)
print(s13.to_string(index=False))
print("\n（已发表值：男 30/30/31 与女 30/31/30 省份数；组数 24/25/28 与 23/25/25；"
      "验证人数 794/2366/2175 与 396/584/470）")
s13.to_csv("results/tables/s13_province_groups.csv", index=False)
print("-> results/tables/s13_province_groups.csv\n")

# ======================================================================
# S4:三种 boosting 框架
# ======================================================================
print("=" * 78)
print("Supplementary Table S4")
print("=" * 78)
missing = []
try:
    from xgboost import XGBRegressor
except ImportError:
    missing.append("xgboost")
try:
    from lightgbm import LGBMRegressor
except ImportError:
    missing.append("lightgbm")
if missing:
    print("缺少依赖:" + ", ".join(missing))
    print("装上再跑这一段:  pip install " + " ".join(missing))
    print("(S13 已经算好了,上面的结果可以直接用。)")
    sys.exit(0)

gb_params = config["modeling"]["algorithms"]["gradient_boosting"]
n_est = gb_params["n_estimators"]
depth = gb_params["max_depth"]
lr = gb_params["learning_rate"]
print(f"三个框架统一设置:n_estimators={n_est}, max_depth={depth}, learning_rate={lr}\n")


def imp(m):
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", m)])


models = {
    "GradientBoosting": s11.get_models(config)["GradientBoosting"],
    "XGBoost": imp(XGBRegressor(n_estimators=n_est, max_depth=depth, learning_rate=lr,
                                random_state=seed, n_jobs=1, verbosity=0)),
    "LightGBM": imp(LGBMRegressor(n_estimators=n_est, max_depth=depth, learning_rate=lr,
                                  num_leaves=2 ** depth - 1, random_state=seed,
                                  n_jobs=1, verbose=-1)),
}

rows4 = []
for cutoff, sx, lab in S4_CELLS:
    ids = gl[(gl[f"cutoff_{cutoff}_group"] == "A") & (gl["sex"] == sx)]["athlete_id"].tolist()
    feats = s11.build_features_for_group_a(
        ids, rec_100m, athletes, cutoff, config["features"]["min_records_for_slope"])
    target = rec_100m.groupby("athlete_id")["time_raw"].min()
    feats["target"] = target.reindex(feats.index)
    feats = feats.dropna(subset=["target"])
    cols = [c for c in s11.get_variant_cols(registry, VARIANT) if c in feats.columns]
    X = feats[cols].values.astype(np.float64)
    y = feats["target"].values.astype(np.float64)
    row = dict(cutoff=cutoff, sex=lab, n=len(y))
    for name, mdl in models.items():
        r = random_cv(mdl, X, y, n_folds, n_rep, seed, n_jobs=n_jobs)
        row[name] = f"{r['R2_mean']:.3f} [{r['R2_ci_lo']:.3f}, {r['R2_ci_hi']:.3f}]"
        row[name + "_raw"] = r["R2_mean"]
        print(f"  {lab} {cutoff}  {name:<17} n={len(y):<5} {row[name]}", flush=True)
    rows4.append(row)

s4 = pd.DataFrame(rows4)
s4.to_csv("results/tables/s4_algorithm_comparison.csv", index=False)
print("\n-> results/tables/s4_algorithm_comparison.csv")

# 自检:GB 那列必须与主表逐位相同
print("\n自检:GB 列对主表")
t2 = pd.read_csv("results/tables/table2_group_a_performance.csv")
t2 = t2[(t2.variant == VARIANT) & (t2.model == "GradientBoosting") & (t2.cv_strategy == "random")]
ok = True
for r in rows4:
    pub = t2[(t2.cutoff_age == r["cutoff"]) & (t2.sex == r["sex"])]["R2_mean"].iloc[0]
    d = abs(pub - r["GradientBoosting_raw"])
    flag = "" if d < 1e-9 else "   <-- 对不上"
    if d >= 1e-9:
        ok = False
    print(f"  {r['sex']} {r['cutoff']}: 这里 {r['GradientBoosting_raw']:.6f} | "
          f"主表 {pub:.6f} | 差 {d:.2e}{flag}")
print("\n" + ("GB 列与主表完全一致,S4 可用" if ok else "GB 列与主表不一致,先别用这批数"))
