# -*- coding: utf-8 -*-
"""
s16_sensitivity_dk.py — D / K / R5-M2 sensitivity analyses (BMC revision).
Run from project root:
    python3 s16_sensitivity_dk.py                 # headline cells, 200 reps (fast, ~10-20 min)
    python3 s16_sensitivity_dk.py --reps 500      # paper-final reps
    python3 s16_sensitivity_dk.py --all-cells     # all cutoffs/sexes (slow)
Prints live progress. Writes CSVs incrementally to results/tables/ (safe to Ctrl+C).
Outputs: sensitivity_D_membership.csv, sensitivity_K_followup.csv, sensitivity_R2_by_definition.csv
GB(100,depth3,lr0.1), seed 42, traj_wind features, athlete-disjoint CV.
"""
import argparse, sys, time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import r2_score
from joblib import Parallel, delayed

SEED=42
def log(m): print(m, flush=True)
TRAJ_WIND=["best_time_raw","mean_time_raw","median_time_raw","sd_time_raw","cv_time_raw","worst_time_raw","range_time_raw","q25_time_raw","q75_time_raw",
"best_time_wc","mean_time_wc","median_time_wc","sd_time_wc","cv_time_wc","worst_time_wc","range_time_wc","q25_time_wc","q75_time_wc",
"improvement_slope_raw","improvement_slope_wc","improvement_residual_sd","best_minus_first","best_minus_latest","improvement_rate_annual",
"n_records","n_competitions","n_seasons","career_span_years","age_first_record","age_best_record","records_per_year",
"has_final","n_finals","pct_finals","heats_final_diff","wind_missing_prop","wind_mean","wind_sd","wind_mean_tail","wind_mean_head"]

def gb(): return Pipeline([("imp",SimpleImputer(strategy="median")),
    ("gb",GradientBoostingRegressor(n_estimators=100,max_depth=3,learning_rate=0.1,subsample=1.0,random_state=SEED))])

def random_cv_r2(X,y,n_repeats,n_folds=5,seed=SEED,n_jobs=-1):
    def one(s):
        kf=KFold(n_splits=n_folds,shuffle=True,random_state=s); out=[]
        for tr,te in kf.split(X):
            m=gb(); m.fit(X[tr],y[tr]); out.append(r2_score(y[te],m.predict(X[te])))
        return out
    res=Parallel(n_jobs=n_jobs)(delayed(one)(seed+i) for i in range(n_repeats))
    v=np.array([x for r in res for x in r]); return v.mean(),np.percentile(v,2.5),np.percentile(v,97.5)

def main(reps, all_cells, root="."):
    t0=time.time(); root=Path(root)
    out=root/"results/tables"; out.mkdir(parents=True,exist_ok=True)
    log(f"[setup] loading data from {root.resolve()} ...")
    gl=pd.read_parquet(root/"data/processed/group_labels.parquet")
    rec=pd.read_parquet(root/"data/interim/cleaned_records.parquet")
    rec=rec[rec["event"]=="100m"][["athlete_id","age_at_comp","competition_date"]].dropna(subset=["age_at_comp"])
    last=rec.groupby("athlete_id").agg(last_age=("age_at_comp","max"),last_date=("competition_date","max")).reset_index()
    gl=gl.merge(last,on="athlete_id",how="left")
    cells=[(c,s) for c in [16,17,18] for s in ["M","F"]] if all_cells else [(16,"M"),(18,"M"),(18,"F")]
    log(f"[setup] done ({time.time()-t0:.0f}s). R2 cells: {cells}  reps={reps}")

    D=[]; K=[]; R=[]
    for cut in sorted(set(c for c,_ in [(c,s) for c in [16,17,18] for s in ['M','F']])):
        feats=pd.read_parquet(root/f"data/processed/features_cutoff_{cut}.parquet"); col=f"cutoff_{cut}_group"
        pc=rec[rec["age_at_comp"]>cut].groupby("athlete_id").size()
        for sx in ["M","F"]:
            elig=feats.merge(gl[gl[col].isin(["A","B"])][["athlete_id",col]],on="athlete_id").query("sex==@sx").dropna(subset=["lifetime_pb"])
            if len(elig)<50: continue
            # D membership (fast)
            y=(elig[col]=="A").astype(int).values; X=elig[TRAJ_WIND].values.astype(float)
            pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("lr",LogisticRegression(max_iter=2000))])
            auc=cross_val_score(pipe,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=SEED),scoring="roc_auc")
            D.append(dict(cutoff=cut,sex=sx,n_eligible=len(elig),pct_A=round(100*y.mean(),1),auc_mean=round(auc.mean(),3),auc_std=round(auc.std(),3)))
            log(f"[D] cutoff{cut} {sx}: AUC={auc.mean():.3f}")
            # K followup (fast)
            A=elig[elig[col]=="A"].merge(gl[["athlete_id","last_age","last_date"]],on="athlete_id",how="left")
            A["fu"]=A["last_age"]-cut; A["pcount"]=A["athlete_id"].map(pc).fillna(0)
            K.append(dict(cutoff=cut,sex=sx,nA=len(A),median_fu=round(A["fu"].median(),2),
                fu_q25=round(A["fu"].quantile(.25),2),fu_q75=round(A["fu"].quantile(.75),2),
                pct_ge1yr=round(100*(A["fu"]>=1).mean(),1),pct_ge2yr=round(100*(A["fu"]>=2).mean(),1),
                pct_last_2024plus=round(100*(pd.to_datetime(A["last_date"]).dt.year>=2024).mean(),1)))
            pd.DataFrame(D).to_csv(out/"sensitivity_D_membership.csv",index=False)
            pd.DataFrame(K).to_csv(out/"sensitivity_K_followup.csv",index=False)
            # R2 sensitivity only for selected cells (heavy)
            if (cut,sx) not in cells: continue
            for lab,sub in [("baseline_ge1rec",A),("followup_ge1yr",A[A.fu>=1]),("followup_ge2yr",A[A.fu>=2]),("groupA_ge2recs",A[A.pcount>=2])]:
                if len(sub)<30:
                    R.append(dict(cutoff=cut,sex=sx,definition=lab,n=len(sub),R2=np.nan,ci_lo=np.nan,ci_hi=np.nan)); continue
                log(f"[R2] cutoff{cut} {sx} {lab}: n={len(sub)} running {reps} reps ...")
                ts=time.time(); m,lo,hi=random_cv_r2(sub[TRAJ_WIND].values.astype(float),sub["lifetime_pb"].values,reps)
                R.append(dict(cutoff=cut,sex=sx,definition=lab,n=len(sub),R2=round(m,3),ci_lo=round(lo,3),ci_hi=round(hi,3)))
                log(f"       -> R2={m:.3f} [{lo:.3f},{hi:.3f}]  ({time.time()-ts:.0f}s)")
                pd.DataFrame(R).to_csv(out/"sensitivity_R2_by_definition.csv",index=False)
    log(f"[done] total {time.time()-t0:.0f}s. CSVs in {out}")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--reps",type=int,default=200)
    ap.add_argument("--all-cells",action="store_true"); ap.add_argument("--root",default=".")
    a=ap.parse_args(); main(a.reps,a.all_cells,a.root)
