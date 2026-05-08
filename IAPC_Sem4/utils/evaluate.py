from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nonzero = y_true != 0
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def rank_models(metrics_df: pd.DataFrame) -> pd.DataFrame:

    # ------------------------------------------------------------
    # Base-model metrics dataframe
    # ------------------------------------------------------------

    if "Test_R2" in metrics_df.columns:

        return metrics_df.sort_values(
            [
                "Test_R2",
                "Test_RMSE",
                "Test_MAE",
            ],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Ensemble metrics dataframe
    # ------------------------------------------------------------

    return metrics_df.sort_values(
        [
            "R2",
            "RMSE",
            "MAE",
        ],
        ascending=[False, True, True],
    ).reset_index(drop=True)

def fit_stacked_ensemble(
    predictions: pd.DataFrame,
    model_names: tuple[str, ...],
) -> np.ndarray:

    ensemble_predictions = (
        predictions[list(model_names)]
        .mean(axis=1)
        .to_numpy(dtype=float)
    )

    return ensemble_predictions


def evaluate_fixed_ensembles(
    ticker: str,
    predictions: pd.DataFrame,
    ensemble_groups: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    y_true = predictions["Actual"].to_numpy(dtype=float)

    for ensemble_name, model_names in ensemble_groups.items():
        blended = fit_stacked_ensemble(predictions, model_names)
        rows.append(
            {
                "Ticker": ticker,
                "Ensemble": ensemble_name,
                "Members": " + ".join(model_names),
                "NumModels": len(model_names),
                **regression_metrics(y_true, blended),
            }
        )

    return rank_models(pd.DataFrame(rows)) if rows else pd.DataFrame()
