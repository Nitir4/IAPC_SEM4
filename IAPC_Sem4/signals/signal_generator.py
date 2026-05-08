# signals/signal_generator.py

import pandas as pd
import numpy as np
from config import BUY_THRESHOLD, SELL_THRESHOLD
from signals.confidence import compute_confidence_series
from signals.risk_engine import compute_risk_series

def generate_signals(predictions_dict, df_index, regime_series,
                     volatility_series, atr_series=None):
    """
    Generate BUY/SELL/HOLD signals with confidence and risk.
    
    predictions_dict: {model_name: np.array of predictions}
    df_index: DatetimeIndex matching predictions
    """
    all_preds_df = pd.DataFrame(predictions_dict, index=df_index)

    # Ensemble prediction = mean across models
    all_preds_df['ensemble'] = all_preds_df.mean(axis=1)

    # Raw signal from ensemble prediction
    def raw_signal(pred):
        if regime == 'Bullish':

            if pred > BUY_THRESHOLD:
                signal = 'BUY'
            else:
                signal = 'HOLD'

        elif regime == 'Bearish':

            if pred < SELL_THRESHOLD:
                signal = 'SELL'
            else:
                signal = 'HOLD'

        else:

            signal = 'HOLD'

    all_preds_df['raw_signal'] = all_preds_df['ensemble'].apply(raw_signal)

    # Confidence + Risk
    all_preds_df['Confidence'] = compute_confidence_series(
        all_preds_df.drop(columns=['ensemble', 'raw_signal']),
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