from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import numpy as np

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


def run_ticker(
    ticker: str,
    data_dir: Path,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # ============================================================
    # LOAD + FEATURE ENGINEERING
    # ============================================================

    df = load_data(ticker, data_dir)

    df = create_features(df)

    x, y = build_tabular_frame(df)

    x_train, x_test, y_train, y_test = chronological_split(
        x,
        y,
        config.train_fraction,
    )

    # ============================================================
    # STORAGE
    # ============================================================

    predictions = pd.DataFrame(index=x_test.index)

    metrics_rows: list[dict[str, float | str]] = []

    # ============================================================
    # TRAIN + EVALUATE BASE MODELS
    # ============================================================

    for model_name in MODEL_MAP:

        print(f"[{ticker}] training {model_name}", flush=True)

        estimator = build_estimator(model_name)

        # --------------------------------------------------------
        # FIT
        # --------------------------------------------------------

        estimator.fit(x_train, y_train)

        # --------------------------------------------------------
        # TRAIN PREDICTIONS
        # --------------------------------------------------------

        train_pred = estimator.predict(x_train)

        # --------------------------------------------------------
        # TEST PREDICTIONS
        # --------------------------------------------------------

        test_pred = estimator.predict(x_test)

        predictions[model_name] = test_pred

        # --------------------------------------------------------
        # METRICS
        # --------------------------------------------------------

        train_metrics = regression_metrics(y_train, train_pred)

        test_metrics = regression_metrics(y_test, test_pred)

        # --------------------------------------------------------
        # GENERALIZATION GAP
        # --------------------------------------------------------

        generalization_gap = (
            train_metrics["R2"] - test_metrics["R2"]
        )

        # --------------------------------------------------------
        # OVERFITTING FLAG
        # --------------------------------------------------------

        overfit_flag = (
            "YES"
            if generalization_gap > 0.15
            else "NO"
        )

        # --------------------------------------------------------
        # PRINT DIAGNOSTICS
        # --------------------------------------------------------

        print(
            f"[{ticker}] {model_name} | "
            f"Train R2={train_metrics['R2']:.4f} | "
            f"Test R2={test_metrics['R2']:.4f} | "
            f"Gap={generalization_gap:.4f} | "
            f"Overfit={overfit_flag}",
            flush=True,
        )

        # --------------------------------------------------------
        # STORE METRICS
        # --------------------------------------------------------

        metrics_rows.append(
            {
                "Ticker": ticker,
                "Model": model_name,

                # TRAIN METRICS
                "Train_MAE": train_metrics["MAE"],
                "Train_MAPE": train_metrics["MAPE"],
                "Train_RMSE": train_metrics["RMSE"],
                "Train_R2": train_metrics["R2"],

                # TEST METRICS
                "Test_MAE": test_metrics["MAE"],
                "Test_MAPE": test_metrics["MAPE"],
                "Test_RMSE": test_metrics["RMSE"],
                "Test_R2": test_metrics["R2"],

                # GENERALIZATION
                "Generalization_Gap": generalization_gap,
                "Overfit": overfit_flag,
            }
        )

    # ============================================================
    # NAIVE BASELINE
    # ============================================================

    naive_pred = (
        df["Close"]
        .shift(1)
        .reindex(x_test.index)
        .to_numpy(dtype=float)
    )

    predictions["NaivePreviousClose"] = naive_pred

    predictions["Actual"] = y_test.to_numpy(dtype=float)

    naive_metrics = regression_metrics(y_test, naive_pred)

    metrics_rows.append(
        {
            "Ticker": ticker,
            "Model": "NaivePreviousClose",

            "Train_MAE": np.nan,
            "Train_MAPE": np.nan,
            "Train_RMSE": np.nan,
            "Train_R2": np.nan,

            "Test_MAE": naive_metrics["MAE"],
            "Test_MAPE": naive_metrics["MAPE"],
            "Test_RMSE": naive_metrics["RMSE"],
            "Test_R2": naive_metrics["R2"],

            "Generalization_Gap": np.nan,
            "Overfit": "BASELINE",
        }
    )

    # ============================================================
    # SAFE ENSEMBLES (NO LEAKAGE)
    # ============================================================

    for ensemble_name, model_names in FIXED_ENSEMBLE_GROUPS.items():

        # --------------------------------------------------------
        # SAFE AVERAGING ENSEMBLE
        # --------------------------------------------------------

        predictions[ensemble_name] = (
            predictions[list(model_names)]
            .mean(axis=1)
            .to_numpy(dtype=float)
        )

        ensemble_metrics = regression_metrics(
            y_test,
            predictions[ensemble_name].to_numpy(dtype=float),
        )

        metrics_rows.append(
            {
                "Ticker": ticker,
                "Model": ensemble_name,

                "Train_MAE": np.nan,
                "Train_MAPE": np.nan,
                "Train_RMSE": np.nan,
                "Train_R2": np.nan,

                "Test_MAE": ensemble_metrics["MAE"],
                "Test_MAPE": ensemble_metrics["MAPE"],
                "Test_RMSE": ensemble_metrics["RMSE"],
                "Test_R2": ensemble_metrics["R2"],

                "Generalization_Gap": np.nan,
                "Overfit": "ENSEMBLE",
            }
        )

    # ============================================================
    # FINAL METRICS DATAFRAME
    # ============================================================

    metrics_df = rank_models(
        pd.DataFrame(metrics_rows)
    )

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
