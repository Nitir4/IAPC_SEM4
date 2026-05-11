from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from paper_reproduction.config import ExperimentConfig
# Import the persistent backtester instead of the original one
from paper_reproduction.backtesting_persistent import run_strategy_backtests_persistent
from paper_reproduction.data import load_stock_csv

from paper_reproduction.ensemble import run_ticker
from paper_reproduction.plotting import write_all_plots
from paper_reproduction.signals import generate_signals
from paper_reproduction.validation import prediction_validation_frame, regime_summary, split_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the hybrid ensemble stock forecasting paper with PERSISTENT positions (conservative).")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs_persistent")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "GS", "JPM", "XOM", "TSLA", "SPY"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--look-back", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--buy-threshold", type=float, default=0.0003)
    parser.add_argument("--sell-threshold", type=float, default=-0.0003)
    parser.add_argument("--starting-cash", type=float, default=100_000.0)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument(
        "--arima-refit-every",
        type=int,
        default=20,
        help="Re-estimate ARIMA/ARMA parameters every N walk-forward steps.",
    )
    parser.add_argument(
        "--meta-min-rmse-improvement",
        type=float,
        default=0.01,
        help=(
            "Minimum validation RMSE improvement versus NaiveZero required before "
            "using learned meta weights. Use 0 to disable the gate."
        ),
    )
    parser.add_argument(
        "--meta-l2-strength",
        type=float,
        default=0.0,
        help=(
            "L2 penalty toward equal meta weights. Use values like 1, 5, or 10 "
            "to encourage blending instead of winner-take-all weights."
        ),
    )
    parser.add_argument(
        "--meta-max-weight",
        type=float,
        default=0.70,
        help="Maximum learned Hybrid weight assigned to any single base model.",
    )
    parser.add_argument(
        "--trading-selection-min-margin",
        type=float,
        default=0.02,
        help="Validation final-value margin required before TradingHybrid leaves EqualBlend.",
    )
    parser.add_argument(
        "--trading-selection-min-trades",
        type=int,
        default=4,
        help="Minimum validation trades required for a strategy to be eligible.",
    )
    parser.add_argument(
        "--trading-selection-min-exposure",
        type=float,
        default=5.0,
        help="Minimum validation exposure percent required for a strategy to be eligible.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["log_return", "pct_change", "price_change", "direction"],
        default="log_return",
    )
    parser.add_argument(
        "--stacking-mode",
        choices=["paper_like_in_sample"],
        default="paper_like_in_sample",
        help=(
            "paper_like_in_sample fits the meta-learner on the test predictions before evaluating them."
        ),
    )
    return parser.parse_args()


def write_backtest_summaries(backtests: pd.DataFrame, output_dir: Path) -> None:
    strategy_summary = (
        backtests.groupby("Strategy", as_index=False)
        .agg(
            Tickers=("Ticker", "count"),
            TotalFinalValue=("Final Value", "sum"),
            AvgFinalValue=("Final Value", "mean"),
            AvgReturn=("Total Return %", "mean"),
            AvgAlpha=("Alpha %", "mean"),
            AvgSharpe=("Sharpe Ratio", "mean"),
            AvgMaxDrawdown=("Max Drawdown %", "mean"),
            AvgExposure=("Exposure %", "mean"),
        )
        .sort_values("TotalFinalValue", ascending=False)
    )
    strategy_summary.to_csv(output_dir / "strategy_summary.csv", index=False)

    ticker_winners = backtests.loc[
        backtests.groupby("Ticker")["Final Value"].idxmax()
    ].sort_values("Ticker")
    ticker_winners.to_csv(output_dir / "ticker_winners.csv", index=False)


def main() -> None:
    import numpy as np
    import tensorflow as tf

    np.random.seed(42)
    tf.random.set_seed(42)

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig(
        look_back=args.look_back,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.random_state,
        arima_refit_every=args.arima_refit_every,
        meta_min_rmse_improvement=args.meta_min_rmse_improvement,
        meta_l2_strength=args.meta_l2_strength,
        meta_max_weight=args.meta_max_weight,
        trading_selection_min_margin=args.trading_selection_min_margin,
        trading_selection_min_trades=args.trading_selection_min_trades,
        trading_selection_min_exposure=args.trading_selection_min_exposure,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        starting_cash=args.starting_cash,
        commission=args.commission,
        slippage=args.slippage,
        stacking_mode=args.stacking_mode,
        target_mode=args.target_mode,
    )

    all_metrics: list[pd.DataFrame] = []
    all_backtests: list[pd.DataFrame] = []
    all_validation: list[pd.DataFrame] = []
    for ticker in args.tickers:
        print(f"\n[{ticker}] loading data", flush=True)
        csv_path = args.data_dir / f"{ticker}.csv"
        df = load_stock_csv(csv_path)
        regime_summary(df).to_csv(args.output_dir / f"regime_summary_{ticker}.csv", index=False)
        print(f"[{ticker}] running models with {len(df)} rows", flush=True)
        metrics, predictions = run_ticker(ticker, df, config, target_mode=args.target_mode)

        # Filter out RegimeSwitching for persistent mode
        if "RegimeSwitching" in predictions.columns:
            predictions = predictions.drop(columns=["RegimeSwitching"])
        metrics = metrics[metrics["Model"] != "RegimeSwitching"]

        print(f"[{ticker}] writing outputs", flush=True)
        predictions.to_csv(args.output_dir / f"predictions_{ticker}.csv", index_label="Date")
        validation_df = prediction_validation_frame(predictions, ticker)
        validation_df.to_csv(args.output_dir / f"validation_{ticker}.csv", index=False)
        all_validation.append(validation_df)

        split_df = split_summary(df, int(len(df) * 0.6), int(len(df) * 0.8))
        split_df.to_csv(args.output_dir / f"splits_{ticker}.csv", index=False)

        strategy_columns = [
            column
            for column in predictions.columns
            if column != "Actual" and not column.startswith("Naive")
        ]
        base_models = {"ARIMA", "RandomForest", "LSTM"}
        signal_frames = {
            strategy: generate_signals(
                predictions=predictions,
                market_frame=df,
                buy_threshold=config.buy_threshold,
                sell_threshold=config.sell_threshold,
                ensemble_column=strategy,
                agreement_columns=[
                    column
                    for column in ["ARIMA", "RandomForest", "LSTM"]
                    if column in predictions.columns
                ],
                require_agreement=strategy not in base_models,
                agreement_min_votes=2,
            )
            for strategy in strategy_columns
        }
        strategy_signals = pd.concat(
            [frame.assign(Strategy=strategy) for strategy, frame in signal_frames.items()]
        ).rename_axis("Date")
        strategy_signals.to_csv(args.output_dir / f"signals_{ticker}.csv")

        # Use the persistent backtester
        backtest_rows, equity_curves = run_strategy_backtests_persistent(
            ohlcv_df=df,
            signal_frames=signal_frames,
            starting_cash=config.starting_cash,
            commission=config.commission,
            slippage=config.slippage,
        )
        backtest_rows.insert(0, "Ticker", ticker)
        backtest_rows.to_csv(args.output_dir / f"backtest_{ticker}.csv", index=False)
        equity_curves.insert(0, "Ticker", ticker)
        equity_curves.to_csv(args.output_dir / f"backtest_curve_{ticker}.csv", index=False)
        all_backtests.append(backtest_rows)

        write_all_plots(ticker, predictions, args.output_dir)
        all_metrics.append(metrics)
        print(
            metrics[
                ["Ticker", "Model", "MAE", "RMSE", "R2", "DirectionalAccuracy", "IC"]
            ].to_string(index=False)
        )
        print(
            backtest_rows[
                [
                    "Ticker",
                    "Strategy",
                    "Final Value",
                    "Profit Loss",
                    "Total Return %",
                    "Buy Hold Return %",
                    "Alpha %",
                    "Sharpe Ratio",
                    "Max Drawdown %",
                    "Exposure %",
                ]
            ].to_string(index=False)
        )

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)
    if all_backtests:
        backtests_df = pd.concat(all_backtests, ignore_index=True)
        backtests_df.to_csv(args.output_dir / "backtests.csv", index=False)
        write_backtest_summaries(backtests_df, args.output_dir)
    if all_validation:
        pd.concat(all_validation, ignore_index=True).to_csv(
            args.output_dir / "validation.csv", index=False
        )
    print(f"\nWrote metrics to {args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
