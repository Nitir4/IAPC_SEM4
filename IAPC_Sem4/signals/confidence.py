# signals/confidence.py

import numpy as np

def compute_confidence(predictions_dict, regime, volatility):
    """
    Compute prediction confidence based on:
    - Model agreement (low variance = high confidence)
    - Regime (Volatile = lower confidence)
    - Volatility level
    """
    preds = np.array(list(predictions_dict.values()))
    
    pred_std  = np.std(preds)
    pred_mean = np.mean(preds)

    # Base score from model agreement
    if pred_std < 0.002:
        agreement_score = 3      # high agreement
    elif pred_std < 0.005:
        agreement_score = 2
    else:
        agreement_score = 1      # low agreement

    # Regime penalty
    regime_penalty = 1 if regime == 'Volatile' else 0

    # Volatility penalty
    vol_penalty = 1 if (volatility is not None and volatility > 0.02) else 0

    score = agreement_score - regime_penalty - vol_penalty

    if score >= 3:
        return 'HIGH'
    elif score >= 2:
        return 'MEDIUM'
    else:
        return 'LOW'


def compute_confidence_series(all_preds_df, regime_series, volatility_series):
    """
    Apply confidence computation row by row across a dataframe.
    all_preds_df: DataFrame with one column per model, one row per date
    """
    confidences = []
    for idx in all_preds_df.index:
        preds_dict = all_preds_df.loc[idx].to_dict()
        regime     = regime_series.loc[idx]
        vol        = volatility_series.loc[idx]
        c          = compute_confidence(preds_dict, regime, vol)
        confidences.append(c)
    return confidences