# signals/signal_generator.py

import pandas as pd
import numpy as np
from config import BUY_THRESHOLD, SELL_THRESHOLD
from signals.confidence import compute_confidence_series
from signals.risk_engine import compute_risk_series

def generate_signals(predictions_dict, df_index, regime_series,
                     volatility_series, atr_series=None, ensemble_column=None):
    """
    Generate BUY/SELL/HOLD signals with confidence and risk.
    
    predictions_dict: {model_name: np.array of predictions}
    df_index: DatetimeIndex matching predictions
    ensemble_column: Name of the column to use as the final prediction.
    """
    all_preds_df = pd.DataFrame(predictions_dict, index=df_index)

    if ensemble_column and ensemble_column in all_preds_df.columns:
        all_preds_df['ensemble'] = all_preds_df[ensemble_column]
    else:
        # Ensemble prediction = mean across models (excluding non-model columns)
        base_models = [c for c in all_preds_df.columns if c not in ['Actual', 'target', 'Signal', 'Confidence', 'Risk', 'Regime', 'signal_num', 'conf_num', 'risk_num']]
        all_preds_df['ensemble'] = all_preds_df[base_models].mean(axis=1)

    # Raw signal from ensemble prediction
    def raw_signal(pred):
        if pred > BUY_THRESHOLD:
            return 'BUY'
        elif pred < SELL_THRESHOLD:
            return 'SELL'
        else:
            return 'HOLD'

    all_preds_df['raw_signal'] = all_preds_df['ensemble'].apply(raw_signal)

    # Confidence + Risk
    # Drop non-model columns for confidence calculation
    feature_cols = [c for c in all_preds_df.columns if c not in ['ensemble', 'raw_signal', 'Actual', 'target', 'Signal', 'Confidence', 'Risk', 'Regime', 'signal_num', 'conf_num', 'risk_num']]
    all_preds_df['Confidence'] = compute_confidence_series(
        all_preds_df[feature_cols],
        regime_series, volatility_series)

    all_preds_df['Risk'] = compute_risk_series(
        regime_series, volatility_series, atr_series)

    # Overlay: downgrade BUY in bad conditions
    def final_signal(row):
        if row['raw_signal'] == 'BUY':
            if row['Confidence'] == 'LOW' or row['Risk'] == 'HIGH':
                return 'HOLD'
        if row['raw_signal'] == 'SELL':
            if row['Confidence'] == 'LOW':
                return 'HOLD'
        return row['raw_signal']

    all_preds_df['Signal'] = all_preds_df.apply(final_signal, axis=1)
    all_preds_df['Regime'] = regime_series.values

    signal_map = {'BUY': 1, 'SELL': -1, 'HOLD': 0}
    all_preds_df['signal_num'] = all_preds_df['Signal'].map(signal_map)

    conf_map = {'HIGH': 2, 'MEDIUM': 1, 'LOW': 0}
    all_preds_df['conf_num'] = all_preds_df['Confidence'].map(conf_map)

    risk_map = {'HIGH': 2, 'MEDIUM': 1, 'LOW': 0}
    all_preds_df['risk_num'] = all_preds_df['Risk'].map(risk_map)

    return all_preds_df[['ensemble', 'Signal', 'signal_num',
                      'Confidence', 'conf_num',
                      'Risk', 'risk_num', 'Regime']]
