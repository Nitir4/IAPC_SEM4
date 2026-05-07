from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr


def directional_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    return float(np.mean(true_direction == pred_direction))


def information_coefficient(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.unique(y_pred).size < 2 or np.unique(y_true).size < 2:
        return 0.0
    corr, _ = spearmanr(y_true, y_pred, nan_policy="omit")
    return float(corr) if not np.isnan(corr) else 0.0


def regression_metrics(y_true, y_pred, benchmark=None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or not len(y_true) or not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError('Metrics need aligned nonempty finite forecast and target arrays.')
    nonzero = y_true != 0
    squared_error = float(np.sum((y_true-y_pred)**2))
    zero_error = float(np.sum(y_true**2))
    lag_skill = float('nan')
    if benchmark is not None:
        baseline=np.asarray(benchmark,dtype=float)
        if baseline.shape != y_true.shape or not np.isfinite(baseline).all():
            raise ValueError('Benchmark forecasts must be aligned and finite.')
        baseline_error=float(np.sum((y_true-baseline)**2))
        if baseline_error>0:lag_skill=1.-squared_error/baseline_error
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else float("nan"),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred) if len(y_true)>1 else float('nan'),
        "R2ZeroBenchmark": 1.-squared_error/zero_error if zero_error>0 else float('nan'),
        "MSESkillLag1": lag_skill,
        "DirectionalAccuracy": directional_accuracy(y_true, y_pred),
        "IC": information_coefficient(y_true, y_pred),
    }
