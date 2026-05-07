# Hybrid ensemble stock forecasting

This is the project's original stock-forecasting approach, combining ARMA, Random Forest and LSTM forecasts with a residual ensemble. Published research provides related-method and evaluation references, described in [METHODOLOGY_REVIEW.md](METHODOLOGY_REVIEW.md). The active code remains in `paper_reproduction/` for entrypoint compatibility; it is not presented as a reproduction of published results.

## Current evaluation protocol

The default chronological split is approximately **60% base training / 20% meta training / 20% final test**, with integer rounding. `train_fraction=0.8` reserves the last 20% for testing, and `meta_fraction_of_train=0.25` reserves one quarter of that first 80% for meta training. Both fractions now control the actual split and the exported split report.

Base models first train on the base region to produce meta-region forecasts. The residual meta learner fits only those forecasts and meta-region labels. Base models are then refitted on the first 80% to produce test forecasts; the meta weights stay fixed. Test labels are used for evaluation. The old `paper_like_in_sample` argument is a deprecated alias for `chronological_holdout`; it does not enable fitting on test labels.

Forecasts dated `t` use observations through `t-1`. CSV gaps are filled forward only, leading incomplete OHLCV rows are discarded, and indicator warm-up values remain missing. Training excludes incomplete feature rows/windows. Feature and target scalers fit only the relevant training prefix. Regime selection and signal risk filters use indicators from the previous observation.

## Models and outputs

- The default target is one-step log return. The active ARIMA-labelled component uses ARMA `(2, 0, 2)` on the target series, updates its state after each realized observation, and periodically refits. Older price-level ARIMA helpers remain available but are not used by this runner.
- Random Forest uses lagged technical features and 100 estimators by default. The LSTM uses a 30-observation look-back, two 64-unit layers, Adam/MSE, 20 epochs and batch size 32. These models stay fixed within each forecast region.
- `ResidualHybrid` anchors on ARMA and fits two RF/LSTM residual corrections using least squares, or ridge when `--meta-l2-strength` is positive. This is not the old constrained-weight helper.
- `ConfidenceWeighted` weights agreement around the model median. It is a heuristic, not calibrated forecast uncertainty. `RegimeSwitching` uses a fixed regime-to-model map.
- Forecast comparison includes zero-return, lagged-return and training-prefix historical-mean baselines, plus an equal model blend. `R2ZeroBenchmark` compares squared error with zero forecasts; `MSESkillLag1` compares with lagged-return forecasts. Positive values indicate an improvement over that benchmark. Conventional R2, MAE, RMSE, direction agreement and rank IC are also exported. MAPE is retained for compatibility, but percentage errors near zero returns are unsuitable as the main success measure. A zero prediction no longer counts automatically as an UP directional forecast. Forecast metrics do not establish profitability. Unused legacy selection/gating parameters are identified in the metrics metadata; CLI attempts to change those inactive options are rejected.

The regular backtest treats HOLD as flat. The persistent runner keeps a position through HOLD and exits on SELL. **Default execution is `Open[t]`**, after forming a forecast dated t from observations through t-1. A position carried from the previous day earns the overnight move before any opening trade; an opening entry earns only the open-to-close move. Commission and slippage are a combined proportional execution charge c. All-in entries buy shares for wealth/[price*(1+c)]; exits receive shares*price*(1-c). Both modes account for initial-capital drawdown. Open positions are marked at the final close without forced liquidation.

`--execution-mode prior_close` retains the idealized same-close timing for sensitivity comparisons; it assumes the completed prior close can be used for a fill at that close. Next-open mode is a clearer information clock, but remains a daily-bar fill approximation without liquidity/latency modeling. The net buy-and-hold reference pays the same initial entry charge and shares the terminal mark-to-market convention; a separately named gross reference is also exported. Signal thresholds and agreement filters use simple-return units: log-return forecasts are transformed with expm1 and price-change forecasts are divided by the previous close. Direction classification is excluded from this regression/backtest runner.

## Setup and run

Use Python 3.11 or 3.12 with the repository's TensorFlow dependency pins:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m paper_reproduction.run --data-dir data --output-dir outputs_causal
python -m paper_reproduction.run_persistent --data-dir data --output-dir outputs_persistent_causal
```

A faster training smoke run and a configurable split:

```bash
python -m paper_reproduction.run --data-dir data --output-dir outputs_causal --tickers AAPL --epochs 1
python -m paper_reproduction.run --data-dir data --output-dir outputs_causal --train-fraction 0.75 --meta-fraction-of-train 0.3333333333333333
```

GPU setup remains available through `requirements-gpu.txt` and `python -m paper_reproduction.check_gpu`. Check `--help` for active options. Runs export metrics, predictions, split reports, signal/backtest CSVs and plots. Existing tracked output folders are historical results from earlier code; regenerate into the new directories above before drawing conclusions from corrected results.

## Verification and preserved code

```bash
pip install pytest
MPLBACKEND=Agg python -m pytest tests -q
```

The methodology checks cover causal preprocessing, feature windows, chronological stacking, configured exports and backtest accounting. The controller regression uses real RF/ARMA with a substituted LSTM; an additional optional test trains a small real TensorFlow LSTM. Neither certifies forecasting accuracy. The dated methodology review records separate real-data training results and their limitations.

The imported `IAPC_Sem4/` directory is preserved as legacy code. The changes documented here apply to `paper_reproduction/`.

## Latest accounting correction

A completed trade has net factor (exit_price/entry_price)*(1-c)/(1+c), not
(exit_price/entry_price)*(1-c)^2. Curve columns Cash and Shares expose the
self-financing ledger. ExecutionPrice is the canonical fill price; legacy
ExecutionClose remains an alias, including in next-open mode. Excess Return vs
Buy Hold % is the raw benchmark return difference; Alpha % remains a compatibility
alias and must not be interpreted as risk-adjusted regression alpha.

Signal requests for missing strategy/agreement columns or invalid votes fail
explicitly. Risk uses lagged ATR/lagged close, with a declared heuristic 2.5%
ATR-fraction threshold. This fixes price-unit dependence; it does not calibrate
risk or confidence statistically. Agreement aliases the historical Confidence
label, which remains heuristic.
