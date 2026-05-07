# Hybrid Ensemble Stock Forecasting Reproduction

This project recreates the methodology described in:

`Hybrid Ensemble Model Approaches for Stock Price Forecasting Using LSTM, Random Forest, ARIMA, and Linear Regression as Meta-Learner`

The paper does not publish source code, so this implementation follows the described methodology as closely as possible and documents the assumptions needed to make the pipeline executable.

## What Is Implemented

- Chronological 80/20 train/test split for each stock CSV.
- Missing value handling with interpolation and forward/back fill.
- LSTM with a 30-day look-back window, two stacked 64-unit LSTM layers, Adam, MSE, 20 epochs, batch size 32.
- Random Forest with 100 estimators and `random_state=42`.
- Technical indicators for Random Forest: SMA, MACD, volatility, and volume averages.
- ARIMA with order `(4, d, 0)`, where `d` is chosen with an Augmented Dickey-Fuller stationarity test.
- Linear Regression meta-learner trained on base model predictions.
- MAE, RMSE, MAPE, and R2 metrics.
- Actual-vs-predicted plots for the hybrid model.

## Replication Assumptions

The paper says Linear Regression is used as a meta-learner but does not specify the exact stacking split protocol. This implementation keeps the paper-like version:

`paper_like_in_sample` trains the Linear Regression meta-learner on the same test-period base predictions it evaluates. This is not a valid out-of-sample evaluation, but it is the closest match to the paper's reported behavior and keeps the reproduced hybrid stable and interpretable.

## Setup

Use Python 3.11. TensorFlow support on Python 3.14 is limited, so avoid the system `python` if it points to 3.14.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For NVIDIA GPU support, first make sure `nvidia-smi` works on the host. Then install TensorFlow's CUDA Python extras:

```bash
pip install -r requirements-gpu.txt
python -m paper_reproduction.check_gpu
```

If `nvidia-smi` fails, fix the Fedora NVIDIA driver/kernel module before debugging TensorFlow.

## Run

```bash
python -m paper_reproduction.run --data-dir data --output-dir outputs_paper_like --stacking-mode paper_like_in_sample
```

Useful faster smoke test:

```bash
python -m paper_reproduction.run --data-dir data --output-dir outputs --tickers AAPL --epochs 1 --stacking-mode paper_like_in_sample
```

The run writes:

- `outputs/metrics.csv`
- `outputs/predictions_<TICKER>.csv`
- `outputs/<TICKER>_model_comparison.png`
- `outputs/<TICKER>_residuals.png`
- `outputs/<TICKER>_metric_bars.png`

## Note on IAPC_Sem4 folder

- The `IAPC_Sem4` subfolder has been imported into this repository so it can be used directly.
- The original embedded Git metadata was moved to `IAPC_Sem4/.git_backup_20260507_210520`.
- `IAPC_Sem4/outputs*` and `IAPC_Sem4/__pycache__` remain ignored by `.gitignore`.

If you want `IAPC_Sem4` to remain an independent repository instead, restore its `.git` from the backup and convert it to a submodule or separate repo.
