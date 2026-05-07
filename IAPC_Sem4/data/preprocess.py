from __future__ import annotations

import numpy as np
import pandas as pd

from IAPC_Sem4.config import TABULAR_FEATURE_COLUMNS


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]

    out["SMA_5"] = close.rolling(5, min_periods=1).mean()
    out["SMA_10"] = close.rolling(10, min_periods=1).mean()
    out["SMA_20"] = close.rolling(20, min_periods=1).mean()
    out["SMA_30"] = close.rolling(30, min_periods=1).mean()

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema_12 - ema_26
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

    returns = close.pct_change()
    out["volatility_10"] = returns.rolling(10, min_periods=2).std()
    out["volatility_30"] = returns.rolling(30, min_periods=2).std()
    out["volume_avg_5"] = out["Volume"].rolling(5, min_periods=1).mean()
    out["volume_avg_20"] = out["Volume"].rolling(20, min_periods=1).mean()

    return out.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def build_tabular_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df[TABULAR_FEATURE_COLUMNS].shift(1)
    y = df["Close"]
    valid = x.notna().all(axis=1) & y.notna()
    return x.loc[valid], y.loc[valid]


def chronological_split(
    x: pd.DataFrame,
    y: pd.Series,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_at = int(len(x) * train_fraction)
    return x.iloc[:split_at], x.iloc[split_at:], y.iloc[:split_at], y.iloc[split_at:]

