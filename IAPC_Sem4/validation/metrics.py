# validation/metrics.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    da   = directional_accuracy(y_true, y_pred)

    metrics = {'MAE': round(mae,6), 'RMSE': round(rmse,6),
               'R2': round(r2,4), 'DirectionalAccuracy': round(da,4)}
    if label:
        print(f"\n[{label}]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics

def directional_accuracy(y_true, y_pred):
    """What % of time did we predict the correct direction (up/down)?"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    correct = np.sign(y_true) == np.sign(y_pred)
    return correct.mean()