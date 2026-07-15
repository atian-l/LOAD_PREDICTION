# -*- coding: utf-8 -*-
"""LightGBM booster 文件 IO 规范化。

背景：仓库 `core.autocrlf=true` 且根 `models/boosters/*.txt` 被 git 跟踪、无 `.gitattributes`
保护。git 在 checkout 时把 LightGBM `save_model` 写出的 LF 换行转成 CRLF，导致
`lgb.Booster(model_file=...)` 解析失败（`Model format error, expect a tree here`）。
这是生产根 bundle（v6）既存的污染，非 v8 引入。

本模块统一读入 booster 文本后做 CRLF->LF 规范化，再用 `model_str=` 加载：
- 不修改磁盘上的根文件（v8 约束：不改既有脚本/代码/模型）；
- 对 LF 文件为 no-op，故 v8 自身 save（LF）-> load 往返同样安全；
- 即便 v8/models 将来被 git 跟踪并被 autocrlf 转 CRLF，也能稳健加载。
"""
from __future__ import annotations
import lightgbm as lgb


def load_booster(path: str) -> lgb.Booster:
    """读取 booster 文本，CRLF->LF 规范化后用 model_str 加载（兼容被 autocrlf 污染的文件）。"""
    with open(path, "rb") as f:
        data = f.read()
    data = data.replace(b"\r\n", b"\n")
    return lgb.Booster(model_str=data.decode("utf-8"))
