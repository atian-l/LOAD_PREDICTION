# -*- coding: utf-8 -*-
"""
Temporal Fusion Transformer（TFT）模型与训练工具。

替代 load_pred 的 LightGBM booster / load_pred_tcn 的 TCN：序列到序列建模范式。
  - encoder：过去 H_enc 步 [pred_load + weather + calendar]（observed past，不含 actual，合规#1/#2）
  - decoder：未来 96 步 [weather + calendar + pred_load]（known future，日前可得，合规#4）
  - static：日级日历（is_holiday/is_weekend/month/dow/doy 的 sin/cos）
  - 输出：未来 96 步 residual（actual - MOS_anchor；保留 MOS 收益）或 direct（actual）
  - multi-horizon：一次前向输出 D+1 全天 96 点（用户建议 Method A 的序列版）

TFT 核心机制（Lim et al. 2021）：Variable Selection Network（变量选择）、Gated Residual
Network（门控残差）、Static Covariate Encoder（静态上下文 c_s/c_e/c_h/c_c）、LSTM encoder-decoder、
Interpretable Temporal Self-Attention（causal，decoder 不看未来）。自包含实现，仅依赖 torch。

合规：
  - encoder 历史负荷用 pred_load（外部预测），绝不用 actual（#1/#2）。
  - 残差目标 = actual - anchor（anchor=MOS corrected pred_load；actual 仅作目标，#1）。
  - 训练只用 < VAL_START 数据（#5）；共享 build_features（#5 train/serve 一致）。
  - encoder_len=288(3天) 覆盖 lag_192=2天（#2/#3）；lag_672 仍作逐时刻特征保留。
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# =========================================================================== #
# 基础组件
# =========================================================================== #
class GRN(nn.Module):
    """Gated Residual Network (GRN): FC -> ELU -> FC -> Dropout -> GLU 门控 -> 残差 + LayerNorm。
    可选 context（静态上下文，broadcast 加到输入）-- TFT 用其注入静态信息。"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout: float = 0.1, context_size: int | None = None):
        super().__init__()
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else None
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.ctx = nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Linear(output_size, output_size)  # GLU 门（作用在 fc2 输出上）
        self.ln = nn.LayerNorm(output_size)

    def forward(self, x, context=None):
        # x: [..., input_size]; context: [..., context_size] (broadcastable to x's leading dims)
        h = self.fc1(x)
        if self.ctx is not None and context is not None:
            h = h + self.ctx(context)
        h = self.elu(h)
        h = self.drop(h)
        e = self.fc2(h)                       # [..., output_size]
        g = torch.sigmoid(self.gate(e))       # GLU 门（标准：e ⊙ σ(W·e)）
        h = e * g
        r = x if self.skip is None else self.skip(x)   # 残差对齐到 output_size
        return self.ln(h + r)


class VariableSelectionNetwork(nn.Module):
    """VSN: 对每个时步的 n_vars 个变量做加权选择，输出 [B, T, hidden]。
    逐变量投影 + GRN 计算变量权重 softmax；可选 static context c_s 条件选择。"""

    def __init__(self, n_vars: int, hidden_size: int, dropout: float = 0.1,
                 context_size: int | None = None):
        super().__init__()
        self.n_vars = n_vars
        self.var_proj = nn.Linear(1, hidden_size)  # 每变量单独投影（共享权重）
        self.weight_grn = GRN(n_vars, hidden_size, n_vars, dropout, context_size=context_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, context=None):
        # x: [B, T, n_vars] -> [B, T, hidden]
        v = self.var_proj(x.unsqueeze(-1))  # [B, T, n_vars, hidden]
        w = self.weight_grn(x, context)     # [B, T, n_vars]
        w = self.softmax(w)
        out = (w.unsqueeze(-1) * v).sum(dim=-2)  # [B, T, hidden]
        return out, w


class TemporalAttention(nn.Module):
    """TFT interpretable temporal self-attention：各头独立 QKV，输出平均后单投影。
    causal mask 使 decoder 位置不看未来。"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.h = num_heads
        self.dh = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(self.dh, hidden_size)  # 各头输出同维，平均后投影
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q = self.q(x).reshape(B, T, self.h, self.dh).transpose(1, 2)  # [B,h,T,dh]
        k = self.k(x).reshape(B, T, self.h, self.dh).transpose(1, 2)
        v = self.v(x).reshape(B, T, self.h, self.dh).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        ctx = torch.matmul(attn, v)  # [B,h,T,dh]
        ctx = ctx.mean(dim=1)        # [B,T,dh] interpretable: 各头平均
        return self.out(ctx)         # [B,T,hidden]


# =========================================================================== #
# TFT 主模型
# =========================================================================== #
class TFT(nn.Module):
    """Temporal Fusion Transformer（序列到序列，multi-horizon 输出 96 点）。

    输入：
      x_enc: [B, H_enc, n_feat]  历史时变（pred_load+weather+calendar，observed past）
      x_dec: [B, 96, n_feat]     未来时变（weather+calendar+pred_load，known future）
      x_static: [B, n_static]    日级日历（static metadata）
    输出：[B, 96] 标量（标准化空间的 residual/direct）

    流程：static->4 上下文(c_s/c_e/c_h/c_c)；enc VSN(c_s)->LSTM(init c_h,c_c)；
    dec VSN(c_s)->LSTM(init c_e)；concat->static enrichment(c_e)->temporal attention(causal)
    ->position GRN->slice decoder->Linear->96。
    """

    def __init__(self, n_feat: int, n_static: int, hidden_size: int = 64,
                 num_heads: int = 2, num_lstm_layers: int = 1, dropout: float = 0.1,
                 encoder_len: int = 288, decoder_len: int = 96):
        super().__init__()
        self.n_feat = n_feat
        self.n_static = n_static
        self.hidden = hidden_size
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len

        # ---- static covariate encoder: static -> 4 上下文 [B, hidden] ----
        self.static_fc = nn.Linear(n_static, hidden_size)
        self.cs_grn = GRN(hidden_size, hidden_size, hidden_size, dropout)
        self.ce_grn = GRN(hidden_size, hidden_size, hidden_size, dropout)
        self.ch_grn = GRN(hidden_size, hidden_size, hidden_size, dropout)
        self.cc_grn = GRN(hidden_size, hidden_size, hidden_size, dropout)

        # ---- variable selection（enc/dec 共享输入变量集 n_feat）----
        self.enc_vsn = VariableSelectionNetwork(n_feat, hidden_size, dropout, context_size=hidden_size)
        self.dec_vsn = VariableSelectionNetwork(n_feat, hidden_size, dropout, context_size=hidden_size)

        # ---- LSTM encoder-decoder ----
        self.lstm_enc = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                                num_layers=num_lstm_layers, batch_first=True)
        self.lstm_dec = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                                num_layers=num_lstm_layers, batch_first=True)
        self.num_lstm = num_lstm_layers

        # ---- static enrichment + temporal attention + position GRN ----
        self.static_enrich = GRN(hidden_size, hidden_size, hidden_size, dropout, context_size=hidden_size)
        self.attention = TemporalAttention(hidden_size, num_heads, dropout)
        self.position_grn = GRN(hidden_size, hidden_size, hidden_size, dropout)
        # 位置编码（可学习）
        self.pos_enc = nn.Parameter(torch.randn(1, encoder_len + decoder_len, hidden_size) * 0.02)

        # ---- 输出头 ----
        self.head = nn.Linear(hidden_size, 1)

        # ---- target 标准化参数（buffer，随 state_dict 持久化；训练时填入，推理反标准化）----
        self.register_buffer("target_mean", torch.tensor(0.0))
        self.register_buffer("target_std", torch.tensor(1.0))

    def _static_contexts(self, x_static):
        s = self.static_fc(x_static)  # [B, hidden]
        c_s = self.cs_grn(s)
        c_e = self.ce_grn(s)
        c_h = self.ch_grn(s)
        c_c = self.cc_grn(s)
        return c_s, c_e, c_h, c_c

    def _causal_mask(self, T, device):
        return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

    def forward(self, x_enc, x_dec, x_static):
        B = x_enc.shape[0]
        c_s, c_e, c_h, c_c = self._static_contexts(x_static)  # [B,hidden]

        # variable selection（context c_s broadcast 到时序）
        enc_sel, _ = self.enc_vsn(x_enc, c_s.unsqueeze(1))  # [B,H_enc,hidden]
        dec_sel, _ = self.dec_vsn(x_dec, c_s.unsqueeze(1))  # [B,96,hidden]

        # LSTM encoder（init from c_h, c_c）
        h0 = c_h.unsqueeze(0).repeat(self.num_lstm, 1, 1)  # [num_lstm,B,hidden]
        c0 = c_c.unsqueeze(0).repeat(self.num_lstm, 1, 1)
        enc_out, _ = self.lstm_enc(enc_sel, (h0, c0))      # [B,H_enc,hidden]

        # LSTM decoder（init from c_e）
        he = c_e.unsqueeze(0).repeat(self.num_lstm, 1, 1)
        ce = c_e.unsqueeze(0).repeat(self.num_lstm, 1, 1)
        dec_out, _ = self.lstm_dec(dec_sel, (he, ce))      # [B,96,hidden]

        # concat + position encoding
        seq = torch.cat([enc_out, dec_out], dim=1)         # [B,H_enc+96,hidden]
        seq = seq + self.pos_enc[:, :seq.shape[1]]

        # static enrichment（context c_e）
        seq = self.static_enrich(seq, c_e.unsqueeze(1))

        # temporal self-attention（causal）
        mask = self._causal_mask(seq.shape[1], seq.device)
        seq = seq + self.attention(seq, mask)              # 残差注意力

        # position-wise GRN
        seq = self.position_grn(seq)

        # 取 decoder 段输出 -> 96 标量
        dec_seq = seq[:, self.encoder_len:]                # [B,96,hidden]
        return self.head(dec_seq).squeeze(-1)              # [B,96]


# =========================================================================== #
# 数据构建：按"预测日"组织序列样本
# =========================================================================== #
def build_day_indices(times, usable, encoder_len: int, decoder_len: int = 96):
    """返回可用预测日的 dec_start_idx 列表。
    条件：encoder 段在范围内；decoder(target) 段 96 点全 usable（actual+pred_load 非空）。
    times[0]=00:00（FULL_START），故 idx%96==0 为日界。"""
    n = len(times)
    # 第一个候选 dec_start：需 encoder_len 历史
    start = encoder_len
    # 对齐到日界（96 边界）
    if start % decoder_len != 0:
        start = (start // decoder_len + 1) * decoder_len
    indices = []
    us = np.asarray(usable, dtype=bool)
    for dec_start in range(start, n - decoder_len + 1, decoder_len):
        if us[dec_start:dec_start + decoder_len].all():
            indices.append(dec_start)
    return indices


def _standardize_fit(arr, mask):
    """arr: [T] 或 [T, D]; mask: [T] bool。在 mask=True 行上算均值/标准差（NaN 安全）。
    返回 (mu, sd)：1D 为标量，2D 为 [D] 向量。sd<=1e-8 或非有限置 1.0。"""
    a = np.asarray(arr, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    sub = a[m]  # 1D: [k]; 2D: [k, D]
    with np.errstate(all="ignore"):
        mu = np.nanmean(sub) if a.ndim == 1 else np.nanmean(sub, axis=0)
        sd = np.nanstd(sub) if a.ndim == 1 else np.nanstd(sub, axis=0)
    if a.ndim == 1:
        mu = float(mu) if np.isfinite(mu) else 0.0
        sd = float(sd) if (np.isfinite(sd) and sd > 1e-8) else 1.0
        return mu, sd
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    return mu, sd


def _apply_std(arr, mu, sd):
    a = np.asarray(arr, dtype=np.float64)
    a = np.where(np.isfinite(a), a, mu)  # NaN 填均值
    return (a - mu) / sd


# =========================================================================== #
# 训练
# =========================================================================== #
def _quantile_loss(pred, target, alpha):
    e = target - pred
    return torch.maximum(alpha * e, (alpha - 1.0) * e)


def train_tft(X: pd.DataFrame, y: pd.Series, w_full: np.ndarray, usable: np.ndarray,
              feat_cols: list, static_cols: list, cfg: dict,
              loss_type: str = "regression", alpha: float = 0.5, seed: int = 42,
              epochs: int = 30, device=None, verbose: bool = False) -> tuple:
    """训练一个 TFT 成员。返回 (tft, feat_mean, feat_std)。
    target(y) 已是 residual 或 direct（由调用方决定），训练折内标准化。"""
    if device is None:
        device = get_device(cfg.get("device", "auto"))
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    encoder_len = int(cfg["encoder_len"])
    decoder_len = int(cfg["decoder_len"])
    hidden = int(cfg["hidden_size"])
    n_feat = len(feat_cols)
    n_static = len(static_cols)

    # ---- 样本：预测日 indices（usable 段）----
    times = X.index
    day_idx = build_day_indices(times, usable, encoder_len, decoder_len)
    if len(day_idx) == 0:
        raise RuntimeError("无可用预测日样本（检查 usable/encoder_len）。")
    # 训练用 mask = usable（target 段全 usable 已保证）；样本权重取 target 段均值
    X_arr = X[feat_cols].to_numpy(dtype=np.float64)
    static_arr = X[static_cols].to_numpy(dtype=np.float64)
    y_arr = y.reindex(times).to_numpy(dtype=np.float64)
    w_arr = np.asarray(w_full, dtype=np.float64)

    # ---- 标准化（feat: 训练样本列均值/标准差；target: 训练样本均值/标准差）----
    # feat 标准化用所有 usable 行
    usable_rows = np.asarray(usable, dtype=bool)
    feat_mean, feat_std = _standardize_fit(X_arr, usable_rows)
    # target 标准化用 usable 行
    y_mu, y_sd = _standardize_fit(y_arr, usable_rows)
    # 预构建标准化数组（full）
    Xs = _apply_std(X_arr, feat_mean, feat_std)          # [T, n_feat]
    ys = (np.where(np.isfinite(y_arr), y_arr, y_mu) - y_mu) / y_sd  # [T]

    # ---- 构建样本张量 ----
    samples = []
    weights = []
    for ds in day_idx:
        enc = Xs[ds - encoder_len:ds]                    # [H_enc, n_feat]
        dec = Xs[ds:ds + decoder_len]                    # [96, n_feat]
        st = static_arr[ds]                              # [n_static]（日级恒定，取日首）
        tgt = ys[ds:ds + decoder_len]                    # [96]
        wt = w_arr[ds:ds + decoder_len].mean()           # 该日权重代表
        if not np.isfinite(tgt).all():
            continue
        samples.append((enc, dec, st, tgt))
        weights.append(float(wt) if np.isfinite(wt) else 1.0)
    if not samples:
        raise RuntimeError("无完整样本（target 含 NaN）。")
    weights = np.array(weights, dtype=np.float32)
    weights = weights / (weights.mean() + 1e-8)

    enc_t = torch.tensor(np.stack([s[0] for s in samples]), dtype=torch.float32, device=device)
    dec_t = torch.tensor(np.stack([s[1] for s in samples]), dtype=torch.float32, device=device)
    st_t = torch.tensor(np.stack([s[2] for s in samples]), dtype=torch.float32, device=device)
    tgt_t = torch.tensor(np.stack([s[3] for s in samples]), dtype=torch.float32, device=device)
    w_t = torch.tensor(weights, dtype=torch.float32, device=device)

    # ---- 模型 ----
    model = TFT(n_feat, n_static, hidden, int(cfg["num_heads"]), int(cfg["num_lstm_layers"]),
                float(cfg["dropout"]), encoder_len, decoder_len).to(device)
    model.target_mean.fill_(float(y_mu))
    model.target_std.fill_(float(y_sd))
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]),
                           weight_decay=float(cfg["weight_decay"]))
    scheduler = None
    if cfg.get("lr_schedule") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(epochs, 1), eta_min=float(cfg.get("lr_eta_min", 1e-5)))

    n = len(samples)
    bs = int(cfg["batch_size"])
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xe, xd, xs, yt, ww = enc_t[idx], dec_t[idx], st_t[idx], tgt_t[idx], w_t[idx]
            pred = model(xe, xd, xs)                # [B,96] 标准化空间
            if loss_type == "quantile":
                loss = _quantile_loss(pred, yt, alpha).mean(dim=1)  # 96 点平均 pinball
            else:
                loss = ((pred - yt) ** 2).mean(dim=1)               # 96 点 MSE
            loss = (loss * ww).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            opt.step()
            total += loss.item() * len(idx)
        if scheduler is not None:
            scheduler.step()
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"      [TFT ep {ep+1}/{epochs}] loss={total/n:.4f}")
    model.eval()
    return model, feat_mean, feat_std


# =========================================================================== #
# 推理：对 full range 每个预测日前向，拼回逐时刻 [T]
# =========================================================================== #
@torch.no_grad()
def predict_tft(tft: TFT, X: pd.DataFrame, feat_cols: list, static_cols: list,
                feat_mean: np.ndarray, feat_std: np.ndarray,
                device, usable: np.ndarray | None = None) -> np.ndarray:
    """返回 full 长度的逐时刻预测（原始量纲，残差/直接空间）。
    预测日段填值，前 encoder_len 段为 NaN。"""
    tft.eval()
    encoder_len = tft.encoder_len
    decoder_len = tft.decoder_len
    times = X.index
    n = len(times)
    X_arr = X[feat_cols].to_numpy(dtype=np.float64)
    static_arr = X[static_cols].to_numpy(dtype=np.float64)
    Xs = _apply_std(X_arr, feat_mean, feat_std)

    out = np.full(n, np.nan, dtype=np.float64)
    # 预测日：从 encoder_len 到 n-dec，步 96，对齐日界
    start = encoder_len
    if start % decoder_len != 0:
        start = (start // decoder_len + 1) * decoder_len
    ds_list = list(range(start, n - decoder_len + 1, decoder_len))
    bs = 64
    for i in range(0, len(ds_list), bs):
        batch = ds_list[i:i + bs]
        encs, decs, sts = [], [], []
        for ds in batch:
            encs.append(Xs[ds - encoder_len:ds])
            decs.append(Xs[ds:ds + decoder_len])
            sts.append(static_arr[ds])
        xe = torch.tensor(np.stack(encs), dtype=torch.float32, device=device)
        xd = torch.tensor(np.stack(decs), dtype=torch.float32, device=device)
        xs = torch.tensor(np.stack(sts), dtype=torch.float32, device=device)
        pred = tft(xe, xd, xs).cpu().numpy()  # [B,96] 标准化空间
        pred = pred * tft.target_std.item() + tft.target_mean.item()  # 反标准化回原始量纲
        for j, ds in enumerate(batch):
            out[ds:ds + decoder_len] = pred[j]
    return out
