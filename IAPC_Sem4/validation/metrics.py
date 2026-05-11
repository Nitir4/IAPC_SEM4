# validation/metrics.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

def compute_metrics(y_true, y_pred, label=""):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    da   = directional_accuracy(y_true, y_pred)
    ic   = information_coefficient(y_true, y_pred)
    
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if np.any(nonzero) else 0.0

    metrics = {
        'MAE': round(mae, 6),
        'RMSE': round(rmse, 6),
        'MAPE': round(mape, 4),
        'R2': round(r2, 4),
        'DirectionalAccuracy': round(da, 4),
        'IC': round(ic, 4)
    }
    
    if label:
        print(f"\n[{label}]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics

def directional_accuracy(y_true, y_pred):
    """What % of time did we predict the correct direction (up/down)?"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    true_direction = np.where(y_true >= 0, 1, -1)
    pred_direction = np.where(y_pred >= 0, 1, -1)
    return float(np.mean(true_direction == pred_direction))

def information_coefficient(y_true, y_pred):
    """Spearman Rank Correlation between actual and predicted returns."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.unique(y_pred).size < 2 or np.unique(y_true).size < 2:
        return 0.0
    corr, _ = spearmanr(y_true, y_pred, nan_policy="omit")
    return float(corr) if not np.isnan(corr) else 0.0
