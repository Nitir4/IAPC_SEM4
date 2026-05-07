from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from IAPC_Sem4.utils.evaluate import regression_metrics


def model_columns(predictions: pd.DataFrame) -> list[str]:
    return [column for column in predictions.columns if column != "Actual"]


def plot_model_comparison(ticker: str, predictions: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(predictions.index, predictions["Actual"], label="Actual", linewidth=1.7)
    for column in model_columns(predictions):
        ax.plot(predictions.index, predictions[column], label=column, linewidth=1.0, alpha=0.85)
    ax.set_title(f"{ticker} Actual vs Model Predictions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()

    output_path = output_dir / f"{ticker}_model_comparison.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_metric_bars(ticker: str, predictions: pd.DataFrame, output_dir: Path) -> Path:
    rows = []
    for column in model_columns(predictions):
        rows.append({"Model": column, **regression_metrics(predictions["Actual"], predictions[column])})
    metrics = pd.DataFrame(rows).set_index("Model")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, metric in zip(axes.flat, ["MAE", "MAPE", "RMSE", "R2"]):
        metrics[metric].plot(kind="bar", ax=ax)
        ax.set_title(metric)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=25)

    fig.suptitle(f"{ticker} Model Metrics", y=0.995)
    fig.tight_layout()

    output_path = output_dir / f"{ticker}_metric_bars.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_all_plots(ticker: str, predictions: pd.DataFrame, output_dir: Path) -> list[Path]:
    return [
        plot_model_comparison(ticker, predictions, output_dir),
        plot_metric_bars(ticker, predictions, output_dir),
    ]
