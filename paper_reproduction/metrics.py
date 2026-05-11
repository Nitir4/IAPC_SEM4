from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr


def directional_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    true_direction = np.where(y_true >= 0, 1, -1)
    pred_direction = np.where(y_pred >= 0, 1, -1)
    return float(np.mean(true_direction == pred_direction))


def information_coefficient(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.unique(y_pred).size < 2 or np.unique(y_true).size < 2:
        return 0.0
    corr, _ = spearmanr(y_true, y_pred, nan_policy="omit")
    return float(corr) if not np.isnan(corr) else 0.0


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nonzero = y_true != 0
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "DirectionalAccuracy": directional_accuracy(y_true, y_pred),
        "IC": information_coefficient(y_true, y_pred),
    }
