import pandas as pd
from ta import trend, momentum, volatility

from config import (
    SMA_WINDOWS,
    EMA_WINDOWS,
    RSI_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    BOLLINGER_PERIOD,
    ATR_PERIOD,
    VOLATILITY_WINDOW
)


def add_indicators(df):

    df = df.copy()

    # Convert to proper 1D series
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    vol = df['Volume'].squeeze()

    # --- Trend ---
    for w in SMA_WINDOWS:
        sma = trend.SMAIndicator(close, window=w)
        df[f'SMA_{w}'] = sma.sma_indicator().values

    for w in EMA_WINDOWS:
        ema = trend.EMAIndicator(close, window=w)
        df[f'EMA_{w}'] = ema.ema_indicator().values

    # --- MACD ---
    macd_obj = trend.MACD(
        close,
        window_fast=MACD_FAST,
        window_slow=MACD_SLOW,
        window_sign=MACD_SIGNAL
    )

    df['MACD'] = macd_obj.macd().values
    df['MACD_signal'] = macd_obj.macd_signal().values
    df['MACD_hist'] = macd_obj.macd_diff().values

    # --- Momentum ---
    rsi = momentum.RSIIndicator(
        close,
        window=RSI_PERIOD
    )

    df['RSI'] = rsi.rsi().values

    # --- Volatility ---
    bb = volatility.BollingerBands(
        close,
        window=BOLLINGER_PERIOD
    )

    df['BB_upper'] = bb.bollinger_hband().values
    df['BB_lower'] = bb.bollinger_lband().values
    df['BB_width'] = df['BB_upper'] - df['BB_lower']

    atr = volatility.AverageTrueRange(
        high,
        low,
        close,
        window=ATR_PERIOD
    )

    df['ATR'] = atr.average_true_range().values

    df['Volatility'] = (
        df['log_return']
        .rolling(VOLATILITY_WINDOW)
        .std()
    )

    # --- Volume ---
    volume_sma = trend.SMAIndicator(
        vol,
        window=20
    )

    df['Volume_SMA_20'] = (
        volume_sma
        .sma_indicator()
        .values
    )

    df['Volume_ratio'] = (
        vol.values / df['Volume_SMA_20']
    )

    return df