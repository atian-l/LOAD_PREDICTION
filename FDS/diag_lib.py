# -*- coding: utf-8 -*-
"""FDS/diag_lib.py - 诊断共享工具（只读分析，不修改生产代码）。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
OUT = ROOT / "output"
FIG = OUT / "figures"
TBL = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TBL.mkdir(parents=True, exist_ok=True)

# CJK 字体（缺失则回退，不影响数值结果）
for f in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
    try:
        matplotlib.font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


def load_val() -> pd.DataFrame:
    f = OUT / "diag_val.csv"
    return pd.read_csv(f, encoding="utf-8-sig", parse_dates=[0]).set_index("时间") \
        if "时间" in pd.read_csv(f, encoding="utf-8-sig", nrows=0).columns \
        else pd.read_csv(f, encoding="utf-8-sig", parse_dates=["Unnamed: 0"]).set_index("Unnamed: 0")


def load_oof() -> pd.DataFrame:
    f = OUT / "diag_oof.csv"
    head = pd.read_csv(f, encoding="utf-8-sig", nrows=0).columns
    idxcol = head[0]
    return pd.read_csv(f, encoding="utf-8-sig", parse_dates=[idxcol]).set_index(idxcol)


def load_Xval() -> pd.DataFrame:
    f = OUT / "X_val.csv"
    head = pd.read_csv(f, encoding="utf-8-sig", nrows=0).columns
    idxcol = head[0]
    return pd.read_csv(f, encoding="utf-8-sig", parse_dates=[idxcol]).set_index(idxcol)


def metrics(e: np.ndarray, a: np.ndarray | None = None) -> dict:
    """e = error (pred-actual); 若给 a=actual 则算 MAPE。"""
    e = np.asarray(e, float)
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    bias = float(np.mean(e))
    std = float(np.std(e))
    out = {"MAE": mae, "RMSE": rmse, "Bias": bias, "std": std, "N": int(len(e))}
    if a is not None:
        a = np.asarray(a, float)
        out["MAPE"] = float(np.mean(np.abs(e) / np.abs(a) * 100))
    return out


def metrics_by(df: pd.DataFrame, col: str, with_mape: bool = True) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(col, sort=True):
        m = metrics(g["error"].values, g["actual"].values if with_mape else None)
        m[col] = key
        rows.append(m)
    return pd.DataFrame(rows).set_index(col)[["N", "MAE", "RMSE", "Bias", "MAPE", "std"]] \
        if with_mape else pd.DataFrame(rows).set_index(col)[["N", "MAE", "RMSE", "Bias", "std"]]


def save_fig(fig, name: str):
    p = FIG / f"{name}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {p.name}", flush=True)


def save_table(df: pd.DataFrame, name: str):
    p = TBL / f"{name}.csv"
    df.to_csv(p, encoding="utf-8-sig", float_format="%.4f")
    print(f"  [tbl] {p.name}", flush=True)


def acf(x: np.ndarray, nlags: int = 200) -> np.ndarray:
    """自相关函数（有偏估计，除以 N）。"""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    c = np.correlate(x, x, mode="full")[n - 1:]
    c = c / c[0]
    return c[: nlags + 1]


def pacf_yw(x: np.ndarray, nlags: int = 50) -> np.ndarray:
    """Yule-Walker PACF 估计（足够诊断用）。"""
    x = np.asarray(x, float)
    n = len(x)
    ac = acf(x, nlags)[: nlags + 1]
    phi = np.zeros(nlags + 1)
    phi[0] = 1.0
    for k in range(1, nlags + 1):
        # Durbin-Levinson 递推
        num = ac[k] - sum(phi[j] * ac[k - j] for j in range(1, k))
        den = 1.0 - sum(phi[j] * ac[j] for j in range(1, k))
        pk = num / den if den != 0 else 0.0
        new = phi.copy()
        new[k] = pk
        for j in range(1, k):
            new[j] = phi[j] - pk * phi[k - j]
        phi = new
    return phi
