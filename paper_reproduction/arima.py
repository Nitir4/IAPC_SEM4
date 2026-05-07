from __future__ import annotations

import warnings

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def choose_d(close: pd.Series, max_d: int = 2, alpha: float = 0.05) -> int:
    series = close.dropna().astype(float)
    for d in range(max_d + 1):
        candidate = series.diff(d).dropna() if d else series
        if len(candidate) < 20:
            continue
        p_value = adfuller(candidate, autolag="AIC")[1]
        if p_value < alpha:
            return d
    return max_d


def forecast_arima(train_close: pd.Series, steps: int, d: int | None = None) -> tuple[pd.Series, int]:
    if d is None:
        d = choose_d(train_close)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(train_close.astype(float), order=(4, d, 0))
        fit = model.fit()
        forecast = fit.forecast(steps=steps)
    return pd.Series(forecast.to_numpy(), name="ARIMA"), d

