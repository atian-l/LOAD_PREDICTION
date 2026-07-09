# -*- coding: utf-8 -*-
"""FDS/run_all.py - 一键复现诊断体系。

依次运行：ds_prep（数据准备）-> a01~a13（13 项分析）。
全程只读，不修改任何生产代码/模型/训练流程。

运行（从项目根目录）：
    python -m FDS.run_all

说明：各分析模块在 import 时会重新包装 stdout（解决 Windows 控制台 CJK 编码），
若在同一进程内连续 import 会触发 GC 关闭底层缓冲区的 "I/O closed file" 问题，
故本编排器以**子进程**方式逐个运行每个模块，保证隔离与可复现。
"""
from __future__ import annotations
import io, os, subprocess, sys
from pathlib import Path

# 父进程控制台默认 GBK，重新包装为 UTF-8 以正确输出 CJK / 状态符号
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
except Exception:
    pass

FDS_DIR = Path(__file__).resolve().parent
ROOT_DIR = FDS_DIR.parent  # 项目根
OUT_DIR = FDS_DIR / "output"

# 子进程需同时能 import load_pred（项目根）与 diag_lib（FDS 目录）
ENV = os.environ.copy()
ENV["PYTHONPATH"] = os.pathsep.join([str(ROOT_DIR), str(FDS_DIR), ENV.get("PYTHONPATH", "")])
ENV["PYTHONIOENCODING"] = "utf-8"

STAGES = [
    ("ds_prep", "数据准备（加载 v6 只读 -> 生成 val 预测 + 诊断列 + OOF）"),
    ("a01_distribution", "(一) 误差分布"),
    ("a02_temporal", "(二) 误差时间规律"),
    ("a03_load_bins", "(三) 负荷区间"),
    ("a04_weather", "(四) 天气条件"),
    ("a05_heatmaps", "(五) 二维 Heatmap"),
    ("a06_autocorr", "(六) 残差自相关"),
    ("a07_consecutive", "(七) 连续误差"),
    ("a08_ramp", "(八) 爬坡"),
    ("a09_extreme", "(九) 极端天气"),
    ("a10_decomp", "(十) 误差来源拆解"),
    ("a11_feature", "(十一) 特征贡献"),
    ("a12_viz", "(十二) 拟合可视化"),
    ("a13_learnable", "(十三) 剩余可学习信息"),
]


def run_stage(name: str, desc: str) -> bool:
    print(f"\n>>> 运行 {name}  -- {desc}", flush=True)
    script = FDS_DIR / f"{name}.py"
    log_path = OUT_DIR / f"{name}.log"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(FDS_DIR),
            env=ENV,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    ok = proc.returncode == 0
    status = "OK" if ok else f"FAIL(rc={proc.returncode})"
    print(f"<<< {name}: {status}  (日志: {log_path})", flush=True)
    if not ok:
        # 打印日志末尾便于定位
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                tail = f.readlines()[-15:]
            sys.stderr.writelines(tail)
        except Exception:
            pass
    return ok


def main():
    print("=" * 70, flush=True)
    print("FDS 预测诊断体系 - 一键复现", flush=True)
    print(f"项目根: {ROOT_DIR}", flush=True)
    print(f"FDS 目录: {FDS_DIR}", flush=True)
    print("=" * 70, flush=True)

    results = []
    for name, desc in STAGES:
        ok = run_stage(name, desc)
        results.append((name, ok))
        if name == "ds_prep" and not ok:
            # 数据准备失败则后续分析无法进行
            print("\n[FATAL] ds_prep 失败，终止。请检查日志。", flush=True)
            break
        if not ok:
            print(f"[WARN] {name} 失败，继续后续阶段（其依赖数据可能缺失）", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("复现汇总", flush=True)
    print("=" * 70, flush=True)
    n_ok = 0
    for name, ok in results:
        print(f"  {'[OK]' if ok else '[FAIL]'}  {name}", flush=True)
        n_ok += int(ok)
    print(f"\n完成: {n_ok}/{len(STAGES)} 阶段成功", flush=True)
    print(f"诊断报告: {FDS_DIR / 'report.md'}", flush=True)
    print(f"图表目录: {OUT_DIR / 'figures'}", flush=True)
    print(f"数据表目录: {OUT_DIR / 'tables'}", flush=True)
    sys.exit(0 if n_ok == len(STAGES) else 1)


if __name__ == "__main__":
    main()
