from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import regression_metrics


def split_summary(df: pd.DataFrame, base_train_end: int, meta_train_end: int) -> pd.DataFrame:
    rows = [
        ("base_train", df.index[:base_train_end]),
        ("meta_train", df.index[base_train_end:meta_train_end]),
        ("test", df.index[meta_train_end:]),
    ]
    return pd.DataFrame(
        [
            {
                "Split": name,
                "Start": idx[0] if len(idx) else pd.NaT,
                "End": idx[-1] if len(idx) else pd.NaT,
                "Rows": len(idx),
            }
            for name, idx in rows
        ]
    )


def prediction_validation_frame(predictions: pd.DataFrame, ticker: str) -> pd.DataFrame:
    rows = []
    actual = predictions["Actual"].to_numpy(dtype=float)
    for column in predictions.columns:
        if column == "Actual":
            continue
        pred = predictions[column].to_numpy(dtype=float)
        rows.append(
            {
                "Ticker": ticker,
                "Model": column,
                **regression_metrics(actual, pred),
                "PredictionMean": float(np.mean(pred)),
                "PredictionStd": float(np.std(pred)),
                "ActualMean": float(np.mean(actual)),
                "ActualStd": float(np.std(actual)),
            }
        )
    return pd.DataFrame(rows)


def regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "Regime" not in df:
        return pd.DataFrame(columns=["Regime", "Count", "Pct"])
    counts = df["Regime"].value_counts()
    return pd.DataFrame(
        {
            "Regime": counts.index,
            "Count": counts.to_numpy(),
            "Pct": (counts / len(df) * 100).round(2).to_numpy(),
        }
    )
