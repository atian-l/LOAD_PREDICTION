# -*- coding: utf-8 -*-
"""exp56: LSTM 序列模型（用户建议：尝试其他模型/甚至 LSTM）。

假设：序列模型可捕捉残差的自相关（点式 LightGBM 看不到），把 raw 1528.74 压更低。
合规设计（无泄露）：
  - 序列输入 = 过去 H=192 步 (2 天) 的 [pred_load(归一), hour_sin, hour_cos, dow_sin, dow_cos]
    —— pred_load 滞后与日历均可在 T 时刻获得。**序列不含逐时刻气象**（气象去重保留最晚起报，
       含修订 hindsight，序列化会泄露）。
  - 当前步上下文 = 全部 126 维无泄露特征（含 weather[T]、pl_wr、lags）。
  - 目标 = residual = actual - pred_load（残差建模，与 residual 成员同）。
  - 早停 holdout = 训练期内 2025-12-01 ~ 2026-02-28（验证集 eval-only，不参与训练/早停）。
  - 实际负荷仅作目标/评估，绝不作输入。
仅诊断，不写产物。对比 LightGBM raw=1528.74 / 生产=1512.63。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from load_pred import config as C, features as F, train as T


H = 192          # 序列长度（2 天）
HID = 64
EPOCHS = 25
BATCH = 512
LR = 1e-3
HOLDOUT_START = pd.Timestamp("2025-12-01 00:00:00")


class LSTMResidual(nn.Module):
    def __init__(self, seq_dim, ctx_dim, hid=HID):
        super().__init__()
        self.lstm = nn.LSTM(seq_dim, hid, batch_first=True, num_layers=1)
        self.ctx = nn.Sequential(nn.Linear(ctx_dim, hid), nn.ReLU(), nn.Dropout(0.1))
        self.head = nn.Sequential(
            nn.Linear(hid * 2, hid), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hid, 1)
        )

    def forward(self, seq, ctx):
        _, (h, _) = self.lstm(seq)        # h: (1, B, hid)
        hs = h[-1]                         # (B, hid)
        cs = self.ctx(ctx)                 # (B, hid)
        return self.head(torch.cat([hs, cs], dim=1)).squeeze(-1)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    times, X, pred_load, actual = T.build_dataset()
    usable = T.usable_mask(times, pred_load, actual)
    mm = F.MismatchModel().fit(X, usable)
    X = mm.transform(X)
    feat_cols = list(X.columns)
    print(f"features={len(feat_cols)} usable={usable.sum()}", flush=True)

    val = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
           & actual.notna()).values
    # holdout（训练期内，用于早停）
    holdout = usable & np.asarray(times >= HOLDOUT_START)
    train_only = usable & np.asarray(times < HOLDOUT_START)
    print(f"train_only={train_only.sum()} holdout={holdout.sum()} val={val.sum()}", flush=True)

    tidx = pd.DatetimeIndex(times)
    hour = tidx.hour.values
    dow = tidx.dayofweek.values
    hour_sin = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    dow_sin = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    dow_cos = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)

    pl = pred_load.values.astype(np.float32)
    pl_mean = float(np.nanmean(pl[usable]))
    pl_std = float(np.nanstd(pl[usable])) + 1e-6
    pl_n = (pl - pl_mean) / pl_std
    pl_n = np.nan_to_num(pl_n, nan=0.0)

    # 序列特征矩阵 (N, 5): pred_load_norm, hour_sin, hour_cos, dow_sin, dow_cos
    seq_feat = np.stack([pl_n, hour_sin, hour_cos, dow_sin, dow_cos], axis=1).astype(np.float32)

    ctx_full = X[feat_cols].values.astype(np.float32)
    ctx_full = np.nan_to_num(ctx_full, nan=0.0, posinf=0.0, neginf=0.0)
    # 上下文归一化（用训练统计）
    ctx_mean = ctx_full[train_only].mean(0, keepdims=True)
    ctx_std = ctx_full[train_only].std(0, keepdims=True) + 1e-6
    ctx_n = (ctx_full - ctx_mean) / ctx_std

    y_res = (actual - pred_load).values.astype(np.float32)  # 残差目标

    def build_seqs(mask):
        idxs = np.where(mask)[0]
        idxs = idxs[idxs >= H]  # 需 H 历史
        seqs = np.empty((len(idxs), H, 5), dtype=np.float32)
        for k, i in enumerate(idxs):
            seqs[k] = seq_feat[i - H:i]
        return seqs, idxs

    print("building train/holdout/val sequences ...", flush=True)
    s_tr, i_tr = build_seqs(train_only)
    s_hd, i_hd = build_seqs(holdout)
    s_va, i_va = build_seqs(val)
    print(f"  train seqs={len(i_tr)} holdout={len(i_hd)} val={len(i_va)}", flush=True)

    ctx_tr = torch.from_numpy(ctx_n[i_tr]);  y_tr = torch.from_numpy(y_res[i_tr])
    ctx_hd = torch.from_numpy(ctx_n[i_hd]);  y_hd = torch.from_numpy(y_res[i_hd])
    ctx_va = torch.from_numpy(ctx_n[i_va]);  pl_va = pl[i_va];  act_va = actual.values[i_va]
    s_tr_t = torch.from_numpy(s_tr); s_hd_t = torch.from_numpy(s_hd); s_va_t = torch.from_numpy(s_va)

    model = LSTMResidual(5, len(feat_cols))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.SmoothL1Loss()
    n = len(i_tr)
    best_hd = 1e18; best_state = None; bad = 0
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(n)
        tot = 0.0
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            sb = s_tr_t[idx]; cb = ctx_tr[idx]; yb = y_tr[idx]
            opt.zero_grad()
            pred = model(sb, cb)
            loss = lossf(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        # holdout
        model.eval()
        with torch.no_grad():
            ph = model(s_hd_t, ctx_hd)
            hd_mae = (ph - y_hd).abs().mean().item()
        if hd_mae < best_hd - 1.0:
            best_hd = hd_mae; best_state = {k: v.clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
        print(f"  ep{ep+1:02d} train_loss={tot/n:.1f} holdout_MAE={hd_mae:.1f} best={best_hd:.1f} bad={bad}", flush=True)
        if bad >= 5:
            print("  early stop", flush=True); break

    # val eval
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        rv = model(s_va_t, ctx_va).numpy()
    final = pl_va + rv
    mae = np.abs(final - act_va).mean()
    print(flush=True)
    print(f"LSTM raw val MAE = {mae:.2f}  (LightGBM raw=1528.74, 生产=1512.63)", flush=True)
    # 也报 holdout 上是否过拟合
    with torch.no_grad():
        rvh = model(s_hd_t, ctx_hd).numpy()
    hd_final = pl[i_hd] + rvh
    print(f"LSTM holdout(raw) MAE = {np.abs(hd_final - actual.values[i_hd]).mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
