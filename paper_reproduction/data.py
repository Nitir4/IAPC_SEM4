from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PRICE_COLUMNS = ["Close", "High", "Low", "Open", "Volume"]
VOLATILE_THRESHOLD = 0.02


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
    returns = np.log(close / close.shift(1))

    out["log_return"] = returns
    out["return_lag1"] = returns.shift(1)
    out["return_lag2"] = returns.shift(2)
    out["return_lag3"] = returns.shift(3)
    out["open_to_close"] = out["Open"] / (close + 1e-8)
    out["high_to_close"] = out["High"] / (close + 1e-8)
    out["low_to_close"] = out["Low"] / (close + 1e-8)
    out["intraday_range"] = (out["High"] - out["Low"]) / (close + 1e-8)

    out["SMA_5"] = close.rolling(5, min_periods=1).mean()
    out["SMA_10"] = close.rolling(10, min_periods=1).mean()
    out["SMA_20"] = close.rolling(20, min_periods=1).mean()
    out["SMA_30"] = close.rolling(30, min_periods=1).mean()
    out["SMA_50"] = close.rolling(50, min_periods=1).mean()
    out["EMA_10"] = close.ewm(span=10, adjust=False).mean()
    out["EMA_20"] = close.ewm(span=20, adjust=False).mean()

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema_12 - ema_26
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

    out["volatility_10"] = returns.rolling(10, min_periods=2).std()
    out["volatility_30"] = returns.rolling(30, min_periods=2).std()
    out["Volatility"] = returns.rolling(20, min_periods=2).std()
    out["volume_avg_5"] = out["Volume"].rolling(5, min_periods=1).mean()
    out["volume_avg_20"] = out["Volume"].rolling(20, min_periods=1).mean()
    out["volume_ratio_20"] = out["Volume"] / (out["volume_avg_20"] + 1e-8)
    out["Volume_SMA_20"] = out["volume_avg_20"]
    out["Volume_ratio"] = out["volume_ratio_20"]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    out["RSI_14"] = 100 - (100 / (1 + rs))

    bb_mid = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=2).std().fillna(0)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    out["BB_upper"] = bb_upper
    out["BB_lower"] = bb_lower
    out["BB_width"] = bb_upper - bb_lower
    out["BB_pct_B"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
    out["close_to_SMA20"] = close / (out["SMA_20"] + 1e-8)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            out["High"] - out["Low"],
            (out["High"] - prev_close).abs(),
            (out["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["ATR"] = true_range.rolling(14, min_periods=1).mean()

    out["Regime"] = detect_regime(out)

    return out.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def detect_regime(df: pd.DataFrame) -> pd.Series:
    fast = df["SMA_20"]
    slow = df["SMA_50"]
    volatility = df["Volatility"]
    conditions = [
        (fast > slow) & (volatility <= VOLATILE_THRESHOLD),
        (fast < slow) & (volatility <= VOLATILE_THRESHOLD),
        volatility > VOLATILE_THRESHOLD,
    ]
    return pd.Series(
        np.select(conditions, ["Bullish", "Bearish", "Volatile"], default="Sideways"),
        index=df.index,
        name="Regime",
    )


def build_target(df: pd.DataFrame, mode: str = "log_return") -> pd.Series:
    close = df["Close"]
    if mode == "log_return":
        return np.log(close / close.shift(1))
    if mode == "pct_change":
        return close.pct_change()
    if mode == "price_change":
        return close.diff()
    if mode == "direction":
        return np.sign(np.log(close / close.shift(1)))
    raise ValueError(f"Unknown target mode: {mode}")


def random_forest_frame(df: pd.DataFrame, target_mode: str = "log_return") -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = [
        "open_to_close",
        "high_to_close",
        "low_to_close",
        "intraday_range",
        "MACD",
        "MACD_signal",
        "volatility_10",
        "volatility_30",
        "volume_ratio_20",
        "RSI_14",
        "BB_pct_B",
        "close_to_SMA20",
        "return_lag1",
        "return_lag2",
        "return_lag3",
    ]
    x = df[feature_columns].shift(1)
    y = build_target(df, mode=target_mode)
    valid = x.notna().all(axis=1) & y.notna()
    return x.loc[valid], y.loc[valid]


def lstm_feature_columns() -> list[str]:
    return [
        "open_to_close",
        "high_to_close",
        "low_to_close",
        "intraday_range",
        "log_return",
        "return_lag1",
        "return_lag2",
        "return_lag3",
        "RSI_14",
        "BB_pct_B",
        "close_to_SMA20",
        "volatility_10",
        "volatility_30",
        "volume_ratio_20",
    ]
