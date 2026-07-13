# -*- coding: utf-8 -*-
"""
TCN（Temporal Convolutional Network）模型与训练工具。

替代 load_pred 的 LightGBM booster：因果膨胀卷积捕捉时序上下文，输出逐时刻预测。
  - 因果卷积：output[t] 仅依赖 input[<=t]，无未来信息（满足 Inviolable Constraints #5）。
  - 膨胀卷积：dilations=[1,2,4,8]，感受野 RF=1+(k-1)·Σdilations（默认 91 步 ≈ 23h）。
  - 残差块：2 个因果卷积 + 残差连接 + ReLU + Dropout；1×1 卷积头输出标量/时刻。

训练：滑窗 mini-batch（seq_len 窗口、stride 步长），逐时刻加权损失
  （regression=MSE，quantile=pinball）。NaN 用训练期列均值填充——PyTorch 卷积不支持 NaN，
  此为输入预处理（不改变特征定义、无泄露；NaN 位置由历史可得性决定，填常数不引入未来信息）。

推理（predict_tcn）：全序列因果前向，分块（块间 RF 重叠）避免长序列 OOM。
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ---------------- 感受野 ---------------- #
def compute_receptive_field(num_channels: list[int], kernel_size: int) -> int:
    """RF = 1 + (k-1)·Σ dilations，dilations=[1,2,4,...,2^(L-1)]。"""
    dilations = [2 ** i for i in range(len(num_channels))]
    return 1 + (kernel_size - 1) * sum(dilations)


# ---------------- 模型 ---------------- #
class TemporalBlock(nn.Module):
    """单残差块：2 个因果膨胀卷积 + ReLU + Dropout + 残差连接。"""

    def __init__(self, n_in: int, n_out: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        # 对称 pad=(k-1)*dilation，前向裁掉右侧 (k-1)*dilation 实现因果（仅左填充）
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.res = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None

    def _causal(self, x: torch.Tensor) -> torch.Tensor:
        """裁掉右侧 (k-1)*dilation（未来信息），使卷积因果。"""
        return x[..., : -(self.kernel_size - 1) * self.dilation]

    def forward(self, x):  # [B, C, T]
        o = self.drop(self.relu(self._causal(self.conv1(x))))
        o = self.drop(self.relu(self._causal(self.conv2(o))))
        r = x if self.res is None else self.res(x)
        return self.relu(o + r)


class TCN(nn.Module):
    """因果膨胀卷积集成成员：[B, C, T] -> [B, T]（逐时刻标量预测，标准化空间）。

    输入特征与目标均在训练折内标准化（神经网不像树模型对尺度不敏感；raw pred_load~6e4
    直接送入卷积会使 4 层激活爆炸、输出层无法学到正确量级）。target_mean/target_std 为
    该成员目标（direct=actual / residual=actual-anchor）的标准化参数，推理时反标准化。
    """

    def __init__(self, n_features: int, num_channels: list[int], kernel_size: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.num_channels = list(num_channels)
        self.kernel_size = kernel_size
        self.dropout = dropout
        blocks = []
        in_ch = n_features
        for i, out_ch in enumerate(num_channels):
            blocks.append(TemporalBlock(in_ch, out_ch, kernel_size, 2 ** i, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*blocks)
        self.head = nn.Conv1d(in_ch, 1, 1)  # 逐时刻标量（标准化空间）
        # 目标标准化参数（训练后由 train_tcn 设置；推理时反标准化）。作为 buffer 随 state_dict 持久化。
        self.register_buffer("target_mean", torch.tensor(0.0))
        self.register_buffer("target_std", torch.tensor(1.0))

    @property
    def receptive_field(self) -> int:
        return compute_receptive_field(self.num_channels, self.kernel_size)

    def forward(self, x):  # [B, C, T] -> [B, T]（标准化空间预测）
        return self.head(self.network(x)).squeeze(1)


# ---------------- 损失 ---------------- #
def _per_sample_loss(pred: torch.Tensor, y: torch.Tensor, loss_type: str, alpha: float) -> torch.Tensor:
    # pred, y: [B, T]
    if loss_type == "quantile":
        e = y - pred  # pinball: max(alpha*e, (alpha-1)*e)
        return torch.where(e >= 0, alpha * e, (alpha - 1.0) * e)
    return (pred - y) ** 2  # regression: MSE


def _weighted_loss(pred, y, w, loss_type, alpha):
    l = _per_sample_loss(pred, y, loss_type, alpha)  # [B, T]
    wsum = w.sum().clamp_min(1e-8)
    return (l * w).sum() / wsum


# ---------------- 连续段与滑窗 ---------------- #
def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """返回 mask 中连续 True 段的 [start, end) 区间（行索引连续=时间连续，因 full_time_index 规整）。"""
    runs = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _extract_windows(X, y, w, mask, seq_len, stride):
    """从 usable 连续段中提取滑窗。返回三个 list（逐窗口的 [C,seq_len]/[seq_len]/[seq_len]）。"""
    wins_X, wins_y, wins_w = [], [], []
    for s, e in _contiguous_runs(mask):
        if e - s < seq_len:
            continue
        for start in range(s, e - seq_len + 1, stride):
            seg = slice(start, start + seq_len)
            wins_X.append(X[seg].T)   # [C, seq_len]
            wins_y.append(y[seg])     # [seq_len]
            wins_w.append(w[seg])     # [seq_len]
    return wins_X, wins_y, wins_w


# ---------------- 训练 ---------------- #
def train_tcn(X, y, w_full, usable, feat_cols, cfg, loss_type, alpha, seed, device,
              epochs: int | None = None, verbose=False):
    """
    训练单个 TCN 成员。

    参数
    ----
    X : 全量特征 DataFrame（按 full_time_index，含 usable 与非 usable 行）
    y : 目标 Series（direct=actual，residual=actual-anchor），索引与 X 对齐
    w_full : 全长权重数组（与 X 同长；usable 段为时间×负荷权重，其余 0）
    usable : bool mask（训练折内可用点；仅此段参与训练）
    feat_cols : 特征列顺序（与 model.feature_cols 一致）
    cfg : TRAIN_CONFIG
    loss_type : "regression" | "quantile"
    alpha : quantile alpha（regression 时未用）
    seed : 随机种子
    device : torch.device
    epochs : 训练 epoch 数（None 时取 cfg["best_it_fixed"]；与 LightGBM 的 nit=best_it 对偶）

    返回 (TCN, feat_mean, feat_std)。feat_mean/feat_std 为训练折内特征列均值/标准差
    （特征标准化 + NaN 填充用，由调用方统一存一份）；目标标准化参数存于 TCN 缓冲区。
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    n_feat = len(feat_cols)
    # ---- 特征标准化参数（训练折内）----
    Xus = X[usable][feat_cols].to_numpy(dtype=np.float64)
    feat_mean = np.nanmean(Xus, axis=0)
    feat_mean = np.where(np.isfinite(feat_mean), feat_mean, 0.0).astype(np.float32)
    feat_std = np.nanstd(Xus, axis=0)
    feat_std = np.where(np.isfinite(feat_std) & (feat_std > 1e-8), feat_std, 1.0).astype(np.float32)
    # ---- 目标标准化参数（训练折内；direct=actual / residual=actual-anchor，各成员不同）----
    y_us = y.reindex(X.index)[usable].to_numpy(dtype=np.float64)
    target_mean = float(np.nanmean(y_us))
    target_std = float(np.nanstd(y_us))
    if not np.isfinite(target_mean):
        target_mean = 0.0
    if not np.isfinite(target_std) or target_std < 1e-8:
        target_std = 1.0

    X_arr = X[feat_cols].to_numpy(dtype=np.float32)            # [T, C]
    y_arr = y.reindex(X.index).to_numpy(dtype=np.float32)      # [T]
    w_arr = np.asarray(w_full, dtype=np.float32)               # [T]
    usable_np = np.asarray(usable, dtype=bool)

    # ---- 标准化（输入预处理，无泄露：参数仅来自训练折）----
    # 特征：NaN 先填 feat_mean -> 标准化后恰为 0（即该特征均值，中性值）
    X_filled = np.where(np.isnan(X_arr), feat_mean[None, :], X_arr)
    X_norm = ((X_filled - feat_mean[None, :]) / feat_std[None, :]).astype(np.float32)
    # 目标标准化
    y_norm = ((y_arr - target_mean) / target_std).astype(np.float32)

    seq_len = int(cfg["seq_len"])
    stride = int(cfg["stride"])
    rf = compute_receptive_field(cfg["num_channels"], cfg["kernel_size"])
    if seq_len <= rf:
        seq_len = rf + 32  # 保证窗口 > 感受野

    wins_X, wins_y, wins_w = _extract_windows(X_norm, y_norm, w_arr, usable_np, seq_len, stride)
    if not wins_X:
        raise RuntimeError("无可用训练窗口（连续段长度 < seq_len）；请减小 seq_len 或检查 usable。")
    n_win = len(wins_X)
    # 损失掩码：窗口前 rf 步上下文不完整（左侧零填充），不计入损失
    loss_mask = np.ones(seq_len, dtype=bool)
    loss_mask[:rf] = False

    Xb = torch.from_numpy(np.stack(wins_X)).to(device)   # [n_win, C, seq_len]
    yb = torch.from_numpy(np.stack(wins_y)).to(device)   # [n_win, seq_len]
    wb = torch.from_numpy(np.stack(wins_w)).to(device)   # [n_win, seq_len]
    lm = torch.from_numpy(loss_mask).to(device)          # [seq_len]

    model = TCN(n_feat, cfg["num_channels"], cfg["kernel_size"], cfg["dropout"]).to(device)
    with torch.no_grad():
        model.target_mean.fill_(target_mean)
        model.target_std.fill_(target_std)
    epochs = int(cfg["best_it_fixed"]) if epochs is None else int(epochs)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]),
                           weight_decay=float(cfg["weight_decay"]))
    # LR 调度器（Tier1 调优）：cosine 退火，lr 从 learning_rate 衰减到 lr_eta_min
    sched_name = str(cfg.get("lr_schedule", "none")).lower()
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=float(cfg.get("lr_eta_min", 1e-5)))
    else:
        scheduler = None
    batch_size = int(cfg["batch_size"])
    grad_clip = float(cfg.get("grad_clip", 5.0))

    idx = np.arange(n_win)
    for ep in range(epochs):
        rng.shuffle(idx)
        model.train()
        total, nb = 0.0, 0
        for b in range(0, n_win, batch_size):
            bi = idx[b:b + batch_size]
            pred = model(Xb[bi])                        # [B, seq_len] 标准化空间
            loss = _weighted_loss(pred[:, lm], yb[bi][:, lm], wb[bi][:, lm], loss_type, alpha)
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total += loss.item(); nb += 1
        if scheduler is not None:
            scheduler.step()
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"      [tcn ep {ep+1}/{epochs}] loss={total/max(1,nb):.4f} lr={opt.param_groups[0]['lr']:.2e}")
    model.eval()
    return model, feat_mean, feat_std


# ---------------- 推理 ---------------- #
def predict_tcn(model: TCN, X_arr: np.ndarray, feat_mean: np.ndarray, feat_std: np.ndarray,
                device: torch.device, chunk_size: int = 8192) -> np.ndarray:
    """全序列因果前向，返回 [T] numpy（已反标准化到原始量纲）。

    特征用训练折 feat_mean/feat_std 标准化（NaN 填 feat_mean -> 标准化后为 0）；
    模型输出在标准化空间，用 model.target_mean/target_std 反标准化回原始负荷量纲。
    分块处理避免长序列 OOM；每块向前多取 RF 步作为上下文（块间重叠 RF），保证各块预测
    的左侧上下文完整（因果卷积只需向后看）。
    """
    model.eval()
    T = X_arr.shape[0]
    rf = model.receptive_field
    Xf = np.where(np.isnan(X_arr), feat_mean[None, :], X_arr).astype(np.float32)
    Xn = ((Xf - feat_mean[None, :]) / feat_std[None, :]).astype(np.float32)  # [T, C] 标准化
    t_mean = float(model.target_mean)
    t_std = float(model.target_std)
    out = np.empty(T, dtype=np.float32)
    with torch.no_grad():
        s = 0
        while s < T:
            e = min(s + chunk_size, T)
            cs = max(0, s - rf)                        # 多取 RF 步上下文
            chunk = Xn[cs:e]                           # [L, C] 已标准化
            t = torch.from_numpy(chunk.T[None]).to(device).float()  # [1, C, L]
            pred = model(t).squeeze(0).cpu().numpy()   # [L] 标准化空间
            pred = pred * t_std + t_mean               # 反标准化到原始量纲
            off = s - cs
            out[s:e] = pred[off:off + (e - s)]
            s = e
    return out
