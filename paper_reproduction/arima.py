from __future__ import annotations

import warnings

import numpy as np
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


def rolling_arima_forecast(
    close: pd.Series,
    train_end: int,
    predict_start: int,
    predict_end: int,
    d: int | None = None,
    refit_every: int = 5,
) -> tuple[pd.Series, int]:
    """Generate walk-forward one-step ARIMA forecasts over a historical test window."""
    if d is None:
        d = choose_d(close.iloc[:train_end])

    history = close.iloc[:predict_start].astype(float).tolist()
    predictions: list[float] = []
    fit = None

    for offset, actual in enumerate(close.iloc[predict_start:predict_end].astype(float)):
        if fit is None or offset % refit_every == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(history, order=(4, d, 0))
                fit = model.fit()

        forecast = fit.forecast(steps=1)
        predictions.append(float(np.asarray(forecast).reshape(-1)[0]))
        history.append(float(actual))
        if fit is not None and offset % refit_every != refit_every - 1:
            fit = fit.append([float(actual)], refit=False)

    return pd.Series(predictions, name="ARIMA"), d


def rolling_arima_return_forecast(
    returns: pd.Series,
    train_end: int,
    predict_start: int,
    predict_end: int,
    order: tuple[int, int, int] = (2, 0, 2),
    refit_every: int = 5,
) -> tuple[pd.Series, int]:
    """Generate walk-forward one-step ARMA forecasts for stationary return targets."""
    history = returns.iloc[:predict_start].dropna().astype(float).tolist()
    predictions: list[float] = []
    fit = None

    for offset, actual in enumerate(returns.iloc[predict_start:predict_end].astype(float)):
        if fit is None or offset % refit_every == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(history, order=order)
                fit = model.fit()

        forecast = fit.forecast(steps=1)
        predictions.append(float(np.asarray(forecast).reshape(-1)[0]))
        if not np.isnan(actual):
            history.append(float(actual))
            if fit is not None and offset % refit_every != refit_every - 1:
                fit = fit.append([float(actual)], refit=False)

    return pd.Series(predictions, name="ARIMA"), order[1]
