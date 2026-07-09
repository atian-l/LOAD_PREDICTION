# -*- coding: utf-8 -*-
"""实验9（快速）：集成组成/alpha_w 扫描，spring25 单折 best_it。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C
from . import data_loader as dl
from . import features as F


def build():
    load_df = dl.load_load_data().set_index(C.COL_TIME)
    times = dl.full_time_index()
    pred_load = load_df[C.COL_PRED_LOAD].reindex(times)
    actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
    weather = dl.load_weather_dedup(run_time=None)
    X = F.build_features(times, pred_load, weather)
    return times, X, pred_load, actual


def tw(times, mask, alpha):
    t = times[mask]; tmin, tmax = t.min(), t.max()
    if tmax == tmin: return np.ones(len(t))
    return (1.0 + alpha * (t - tmin).total_seconds() / (tmax - tmin).total_seconds()).values.astype(float)


def _m(p, a):
    err = p - a
    return np.mean(np.abs(err)), (1 - np.sum(err**2)/np.sum((a-a.mean())**2)), np.mean(err)


def run(times, X, pred_load, actual, feat_cols, usable, alpha_w, seeds, qalphas, lr, nl, md):
    y_dir = actual; y_res = actual - pred_load
    base = dict(metric=["mae","rmse"], learning_rate=lr, num_leaves=nl, min_data_in_leaf=md,
                lambda_l2=1.0, feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=1,
                verbose=-1, force_col_wise=True, num_iterations=8000, early_stopping_rounds=300)
    pp = dict(base); pp.pop("num_iterations"); pp.pop("early_stopping_rounds")
    te=pd.Timestamp("2025-02-28"); vs=pd.Timestamp("2025-03-01"); ve=pd.Timestamp("2025-05-31")
    ftr=usable&(times<=te); fva=usable&(times>=vs)&(times<=ve)
    dtr=lgb.Dataset(X[ftr][feat_cols],label=y_dir[ftr].values,weight=tw(times,ftr,alpha_w))
    dva=lgb.Dataset(X[fva][feat_cols],label=y_dir[fva].values,reference=dtr)
    ev={}
    b0=lgb.train({**pp,"objective":"regression","seed":42},dtr,num_boost_round=base["num_iterations"],
                 valid_sets=[dva],valid_names=["va"],
                 callbacks=[lgb.early_stopping(base["early_stopping_rounds"],verbose=False,first_metric_only=True),
                            lgb.record_evaluation(ev)])
    best_it=max(b0.best_iteration,80)
    Xtr=X[usable][feat_cols]; wtr=tw(times,usable,alpha_w)
    raw_sum=np.zeros(len(times)); n=0
    objs=[("regression",None)]+[("quantile",q) for q in qalphas]
    for residual in (False,True):
        ytr=(y_res if residual else y_dir)[usable]
        d=lgb.Dataset(Xtr,label=ytr.values,weight=wtr)
        for obj,qa in objs:
            for s in seeds:
                p=dict(pp,objective=obj,seed=s)
                if obj=="quantile": p["alpha"]=qa
                bst=lgb.train(p,d,num_boost_round=int(best_it))
                raw=bst.predict(X[feat_cols])
                raw_sum+=(pred_load.values+raw) if residual else raw
                n+=1
    ens=np.clip(raw_sum/n,0,None)
    vm=((times>=C.VAL_START)&(times<=C.VAL_END)&actual.notna()).values
    mae,r2,bias=_m(ens[vm],actual.values[vm])
    return best_it,mae,r2,bias,n


def main():
    print("building ...")
    times,X,pred_load,actual=build()
    feat_cols=list(X.columns)
    ts0=pd.Timestamp("2023-02-01"); tr_end=pd.Timestamp(C.TRAIN_END)
    usable=((times>=ts0)&(times<=tr_end)&pred_load.notna()&actual.notna()).values
    s5=[42,7,123,2024,99]; s8=s5+[31,256,555]
    cands=[
        ("base",            1.5,s5,[0.5],0.02,127,300),
        ("q045_05_055",     1.5,s5,[0.45,0.5,0.55],0.02,127,300),
        ("seeds8",          1.5,s8,[0.5],0.02,127,300),
        ("aw2.5",           2.5,s5,[0.5],0.02,127,300),
        ("aw2.5_q3",        2.5,s5,[0.45,0.5,0.55],0.02,127,300),
        ("aw2.5_md200_l159",2.5,s5,[0.5],0.02,159,200),
        ("aw3.0",           3.0,s5,[0.5],0.02,127,300),
        ("aw2.0_q3_s8",     2.0,s8,[0.45,0.5,0.55],0.02,127,300),
    ]
    for name,aw,seeds,qa,lr,nl,md in cands:
        best_it,mae,r2,bias,n=run(times,X,pred_load,actual,feat_cols,usable,aw,seeds,qa,lr,nl,md)
        print(f"[{name:18s}] best_it={best_it} n={n}  VAL MAE={mae:.2f}  R2={r2:.4f}  Bias={bias:.0f}")


if __name__=="__main__":
    main()
