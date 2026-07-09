# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A day-ahead (D+1) direct-dispatch load forecasting system for Shandong province. A LightGBM
ensemble corrects an *external* load forecast (`预测直调负荷`) toward the *actual* load
(`实际直调负荷`). Two decoupled entry points: `train.py` (trains + saves + evaluates) and
`predict.py` (loads saved model, infers D+1 with no retraining).

The entire design is built around a set of **data-leakage invariants** (see below). Any change
that lets `实际直调负荷` flow into features, or lets future information reach a prediction
timestamp T, is a regression — even if metrics improve.

## Commands

All commands run from the project root (`load_prediction/`) as modules — never `cd` into
`load_pred/` (relative imports require the package form):

```bash
python -m load_pred.train                      # train ensemble, save to models/, write output/, print val MAE
python -m load_pred.predict --run-date 2026-05-18   # predict D+1 96 points for run date D; default D=today
python -m load_pred.exp                        # walk-forward CV experiment (writes nothing)
python -m load_pred.exp15                      # one of the Agent-Loop experiment scripts (see below)
```

`train.main()` exits non-zero if val MAE ≥ 1500 MW, so it doubles as a pass/fail gate.

Environment: Python 3.14, lightgbm 4.6, pandas 3.0, numpy 2.4, scikit-learn 1.9. There is no
`requirements.txt`, no test suite, no linter config — verification is done by reading val MAE
from `output/evaluation_metrics.txt`.

## Architecture

### The leakage invariants (read multiple files to enforce)

These cross-cut `config.py`, `data_loader.py`, `features.py`, `train.py`, `predict.py`. They
are the load-bearing constraint of the project:

- **Actual load is eval-only.** `实际直调负荷` is read in `data_loader.load_load_data()` and
  consumed *only* in `train.py` (as the regression target `y_dir`, the residual target
  `actual - pred_load`, and the evaluation baseline). `features.py` must never touch it.
- **Lags come from the external forecast only.** `features.pred_load_features` builds
  `PRED_LAGS = [96,192,288,672]` from `预测直调负荷`, never from actual load. `lag_192` (2 days)
  is the mandated minimum lag and must stay present.
- **Weather dedup is the same in all three phases.** `data_loader.load_weather_dedup()` keeps,
  per forecast time, the row with the latest issue time. In predict mode it first filters
  `起报时间 <= run_time` (default run hour **9:00** — deployment runs ~9 AM daily) to simulate
  "only already-issued versions available at runtime." Train mode passes `run_time=None` (all
  history). **9 AM deployment caveat:** the historical weather CSV only has 20:00 issues (the
  20:00 day-D issue covers D+1, but is issued 11h *after* 9 AM). At 9 AM, D+1 weather therefore
  relies on a *morning issue* (e.g. 08:00 day-D) supplied by the deployment feed (user-confirmed
  available). val evaluation uses `run_time=None` → the 20:00 day-D issue as the morning-issue
  proxy (both are "best available D+1 forecast at runtime"), so val MAE=1459 is the
  deployment-realistic baseline. If the morning issue is ever unavailable, predict.py degrades
  gracefully to ~1634 (NaN weather, model falls back to pred_load+calendar; LightGBM handles NaN
  natively). Historical predict backtests must use `--run-hour 21` to reproduce val's
  20:00-D-issue condition (the 9 AM filter excludes it from the 20:00-only CSV).
- **Time boundary.** Training data is capped at `TRAIN_END = 2026-01-31 23:45:00`. The official
  validation window `2026-02-01 ~ 2026-05-19 11:45:00` is *eval-only* — it never trains or
  early-stops. Early stopping uses 3 walk-forward folds (spring/autumn/winter 2025), all inside
  the training period (`config.TRAIN_CONFIG["best_it_folds"]`).
- **Train/predict share one feature builder.** `features.build_features()` is the single
  entry point used by both modes, which prevents train/serve skew.

### Model (`model.py`)

`EnsembleModel` holds N LightGBM boosters. Each member is tagged `is_residual`: residual
members predict `actual - anchor` and are reconstructed as `anchor[T] + raw`; direct members
predict `actual` directly. Final = **median** over members, then shrinkage toward the anchor:
`pred = anchor + λ·(ens − anchor)`. The anchor is the **two-stage MOS** corrected forecast
(`features.MosModel`: `Ridge(actual ~ pred_load + weather + calendar + pl_weather_residual)`,
target=actual which is compliant as a target only) when `config.mos` is set, else raw `pred_load`.
Production config (`config.TRAIN_CONFIG`) yields 40 members = `{regression, quantile(0.45/0.5/0.55)}
× {direct, residual} × 5 seeds`, `best_it_fixed=80` (walk-forward overfits the drifting val - exp44),
recency-weighted samples (`alpha_w`) **plus a load-weighting factor** (`weight_load_gamma=1.0`:
`1 + γ·clip(pl/mean(pl) − 1, −0.5, 1)` on the time weight; input is pred_load only, compliant #2).
OOF-estimated post-hoc corrections applied in `predict_load`: 96-dim quarter hour_bias,
midday drift_corr (β·pl_weather_residual @11-14), and threshold_corr (scenario shifts).
Saved as `models/model_bundle.pkl` + per-member `models/boosters/member_NNN.txt`;
`predict.py` only calls `EnsembleModel.load()`. Current production val MAE = 1445.62 MW (v6).

### Experiment scripts (`exp*.py`)

`exp.py` and `exp2.py`…`exp83.py` are the **Agent Loop** — throwaway hyperparameter/feature
exploration scripts that report val MAE but write no artifacts. They each rebuild the dataset
independently (their own `build()`) and often use different member counts (24, 40) or
recency weights than production. They are *not* part of the production pipeline; treat them as
a log of how the current config was arrived at, not as something to keep in sync. Their
stdout is captured in the top-level `exp*.log` files.

## Gotchas

- **Data file coupling.** `config.py` references `LOAD_CSV = data/direct_load_latest.csv` with
  `COL_PRED_LOAD="预测负荷"` / `COL_ACTUAL_LOAD="实际负荷"` (no `直调`) - these match the file
  currently in `data/`, so `python -m load_pred.train` runs. When the input data is refreshed,
  keep `LOAD_CSV` and the `COL_PRED_LOAD`/`COL_ACTUAL_LOAD` constants in sync with the file/column
  names (or rename the file / columns). The CSVs are UTF-8-BOM, so they are always read with
  `encoding="utf-8-sig"`.
- **Run as a module, not a script.** `python load_pred/train.py` breaks the relative imports
  (`from . import config`). Always use `python -m load_pred.train`.
- **Predict builds on a history window (skew fix).** `predict.predict_d1` builds features on a
  14-day lookback window then extracts the 96 D+1 points - NOT on the 96 points alone. Building on
  96 points alone leaves window>96 rolling and shift(>=96) features all-NaN (train/serve skew);
  the 14-day window (>672+96) makes D+1 features bit-identical to training. Do not revert this.
  `EnsembleModel.save/load` uses only `model_bundle.pkl` and `boosters/member_NNN.txt`.
- `t/` and the `exp*.log` files at the repo root are scratch/experiment output, not inputs.
