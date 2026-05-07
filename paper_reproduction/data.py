from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PRICE_COLUMNS = ["Close", "High", "Low", "Open", "Volume"]


def load_stock_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    missing = {"Date", *PRICE_COLUMNS}.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.sort_values("Date").set_index("Date")
    df[PRICE_COLUMNS] = df[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df[PRICE_COLUMNS] = df[PRICE_COLUMNS].interpolate(method="time")
    df[PRICE_COLUMNS] = df[PRICE_COLUMNS].ffill().bfill()
    return add_technical_features(df)


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
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


def random_forest_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = [
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
        "SMA_5",
        "SMA_10",
        "SMA_20",
        "SMA_30",
        "MACD",
        "MACD_signal",
        "volatility_10",
        "volatility_30",
        "volume_avg_5",
        "volume_avg_20",
    ]
    x = df[feature_columns].shift(1)
    y = df["Close"]
    valid = x.notna().all(axis=1) & y.notna()
    return x.loc[valid], y.loc[valid]


def lstm_feature_columns() -> list[str]:
    return PRICE_COLUMNS.copy()

