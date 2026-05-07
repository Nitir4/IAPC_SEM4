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
from paper_reproduction.data import load_stock_csv
from paper_reproduction.ensemble import run_ticker
from paper_reproduction.plotting import write_all_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the hybrid ensemble stock forecasting paper.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "GS", "JPM", "XOM", "TSLA"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--look-back", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--stacking-mode",
        choices=["paper_like_in_sample"],
        default="paper_like_in_sample",
        help=(
            "paper_like_in_sample fits the meta-learner on the test predictions before evaluating them."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig(
        look_back=args.look_back,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.random_state,
        stacking_mode=args.stacking_mode,
    )

    all_metrics: list[pd.DataFrame] = []
    for ticker in args.tickers:
        print(f"\n[{ticker}] loading data", flush=True)
        csv_path = args.data_dir / f"{ticker}.csv"
        df = load_stock_csv(csv_path)
        print(f"[{ticker}] running models with {len(df)} rows", flush=True)
        metrics, predictions = run_ticker(ticker, df, config)

        print(f"[{ticker}] writing outputs", flush=True)
        predictions.to_csv(args.output_dir / f"predictions_{ticker}.csv", index_label="Date")
        write_all_plots(ticker, predictions, args.output_dir)
        all_metrics.append(metrics)
        print(metrics[["Ticker", "Model", "MAE", "MAPE", "RMSE", "R2"]].to_string(index=False))

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)
    print(f"\nWrote metrics to {args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
