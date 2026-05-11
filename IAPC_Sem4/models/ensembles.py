import numpy as np
import pandas as pd
from scipy.optimize import minimize

def residual_hybrid_predict(meta_x, meta_y, test_x, base_model_name='LinearRegression', residual_model_names=None):
    """
    Fits weights for residual models to predict the error of a base model.
    Adapted from paper_reproduction.
    """
    if residual_model_names is None:
        residual_model_names = [col for col in meta_x.columns if col != base_model_name]
    
    base_meta = meta_x[base_model_name].to_numpy(dtype=float)
    residual_y = meta_y - base_meta
    
    residual_features = np.column_stack([
        meta_x[m].to_numpy(dtype=float) - base_meta for m in residual_model_names
    ])
    
    # Simple Ridge regression for weights
    l2_strength = 0.1
    xtx = residual_features.T @ residual_features + l2_strength * np.eye(residual_features.shape[1])
    weights = np.linalg.solve(xtx, residual_features.T @ residual_y)
    
    base_test = test_x[base_model_name].to_numpy(dtype=float)
    residual_test = np.column_stack([
        test_x[m].to_numpy(dtype=float) - base_test for m in residual_model_names
    ])
    
    pred = base_test + residual_test @ weights
    return pred, weights

def confidence_weighted_predict(test_x, model_names=None, eps=1e-6):
    """
    Weighted average based on distance from the median prediction.
    """
    if model_names is None:
        model_names = list(test_x.columns)
        
    preds = []
    for _, row in test_x[model_names].iterrows():
        values = row.to_numpy(dtype=float)
        median = float(np.median(values))
        distances = np.abs(values - median)
        weights = 1.0 / (distances + eps)
        weights = weights / weights.sum()
        preds.append(float(weights @ values))
    return np.asarray(preds, dtype=float)
