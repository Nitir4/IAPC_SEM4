# data/regime.py

import pandas as pd
import numpy as np
from config import REGIME_SMA_FAST, REGIME_SMA_SLOW, VOLATILE_THRESHOLD

def detect_regime(df):
    """
    Classify each day into a market regime.
    Requires SMA columns already present from indicators.py
    """
    df = df.copy()

    fast_col = f'SMA_{REGIME_SMA_FAST}'
    slow_col = f'SMA_{REGIME_SMA_SLOW}'

    if fast_col not in df.columns or slow_col not in df.columns:
        raise ValueError(f"Missing {fast_col} or {slow_col}. Run add_indicators() first.")

    conditions = [
        (df[fast_col] > df[slow_col]) & (df['Volatility'] <= VOLATILE_THRESHOLD),
        (df[fast_col] < df[slow_col]) & (df['Volatility'] <= VOLATILE_THRESHOLD),
        (df['Volatility'] > VOLATILE_THRESHOLD),
    ]
    choices = ['Bullish', 'Bearish', 'Volatile']

    df['Regime'] = np.select(conditions, choices, default='Sideways')

    return df

def regime_summary(df):
    """Print regime distribution."""
    counts = df['Regime'].value_counts()
    pct = (counts / len(df) * 100).round(1)
    summary = pd.DataFrame({'Count': counts, 'Pct': pct})
    print("\nRegime Distribution:")
    print(summary)
    return summary