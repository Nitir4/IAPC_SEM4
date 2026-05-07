from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

from IAPC_Sem4.config import DATA_DIR, DEFAULT_TICKERS, FIXED_ENSEMBLE_GROUPS, OUTPUT_DIR, ExperimentConfig
from IAPC_Sem4.data.loader import available_tickers, load_data
from IAPC_Sem4.data.preprocess import build_tabular_frame, chronological_split, create_features
from IAPC_Sem4.models import MODEL_MAP
from IAPC_Sem4.utils.evaluate import evaluate_fixed_ensembles, fit_stacked_ensemble, rank_models, regression_metrics
from IAPC_Sem4.utils.plot import write_all_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-style stock experiments for IAPC_Sem4.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def build_estimator(model_name: str):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", MODEL_MAP[model_name]()),
        ]
    )


def run_ticker(ticker: str, data_dir: Path, config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_data(ticker, data_dir)
    df = create_features(df)
    x, y = build_tabular_frame(df)
    x_train, x_test, y_train, y_test = chronological_split(x, y, config.train_fraction)

    predictions = pd.DataFrame(index=x_test.index)
    metrics_rows: list[dict[str, float | str]] = []

    for model_name in MODEL_MAP:
        estimator = build_estimator(model_name)
        estimator.fit(x_train, y_train)
        y_pred = estimator.predict(x_test)
        predictions[model_name] = y_pred
        metrics_rows.append({"Ticker": ticker, "Model": model_name, **regression_metrics(y_test, y_pred)})

    naive_pred = df["Close"].shift(1).reindex(x_test.index).to_numpy(dtype=float)
    predictions["NaivePreviousClose"] = naive_pred
    predictions["Actual"] = y_test.to_numpy(dtype=float)
    metrics_rows.append(
        {
            "Ticker": ticker,
            "Model": "NaivePreviousClose",
            **regression_metrics(y_test, naive_pred),
        }
    )

    for ensemble_name, model_names in FIXED_ENSEMBLE_GROUPS.items():
        predictions[ensemble_name] = fit_stacked_ensemble(predictions, model_names)
        metrics_rows.append(
            {
                "Ticker": ticker,
                "Model": ensemble_name,
                **regression_metrics(y_test, predictions[ensemble_name].to_numpy(dtype=float)),
            }
        )

    metrics_df = rank_models(pd.DataFrame(metrics_rows))
    ensembles_df = evaluate_fixed_ensembles(
        ticker=ticker,
        predictions=predictions,
        ensemble_groups=FIXED_ENSEMBLE_GROUPS,
    )
    return metrics_df, predictions, ensembles_df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_tickers = args.tickers if args.tickers != ["ALL"] else available_tickers(args.data_dir)
    config = ExperimentConfig()

    all_metrics: list[pd.DataFrame] = []
    all_ensembles: list[pd.DataFrame] = []

    for ticker in selected_tickers:
        print(f"\n[{ticker}] loading local CSV and running base models", flush=True)
        metrics_df, predictions, ensembles_df = run_ticker(ticker, args.data_dir, config)
        predictions.to_csv(args.output_dir / f"predictions_{ticker}.csv", index_label="Date")
        write_all_plots(ticker, predictions, args.output_dir)
        metrics_df.to_csv(args.output_dir / f"metrics_{ticker}.csv", index=False)
        if not ensembles_df.empty:
            ensembles_df.to_csv(args.output_dir / f"ensemble_groups_{ticker}.csv", index=False)

        all_metrics.append(metrics_df)
        if not ensembles_df.empty:
            all_ensembles.append(ensembles_df)

        print("[Base models]")
        print(metrics_df.to_string(index=False))
        if not ensembles_df.empty:
            print("\n[Fixed ensemble groups]")
            print(ensembles_df.to_string(index=False))

    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_metrics.to_csv(args.output_dir / "metrics_all_tickers.csv", index=False)

    if all_ensembles:
        combined_ensembles = pd.concat(all_ensembles, ignore_index=True)
        combined_ensembles.to_csv(args.output_dir / "ensemble_groups_all_tickers.csv", index=False)

    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
