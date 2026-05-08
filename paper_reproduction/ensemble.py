from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .arima import forecast_arima
from .config import ExperimentConfig
from .data import random_forest_frame
from .lstm import fit_predict_lstm
from .metrics import regression_metrics


BASE_MODELS = ("ARIMA", "RandomForest", "LSTM")
BASE_MODEL_COLUMNS = list(BASE_MODELS)


def train_rf_predict(
    df: pd.DataFrame,
    train_end: int,
    predict_start: int,
    predict_end: int,
    config: ExperimentConfig,
) -> np.ndarray:
    x, y = random_forest_frame(df)
    train_dates = df.index[:train_end]
    predict_dates = df.index[predict_start:predict_end]

    train_mask = x.index.isin(train_dates)
    predict_mask = x.index.isin(predict_dates)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=config.rf_estimators,
                    random_state=config.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x.loc[train_mask], y.loc[train_mask])
    return model.predict(x.loc[predict_mask])


def base_predictions(
    df: pd.DataFrame,
    train_end: int,
    predict_start: int,
    predict_end: int,
    config: ExperimentConfig,
    label: str,
) -> tuple[pd.DataFrame, int]:
    steps = predict_end - predict_start
    print(f"[{label}] ARIMA forecast for {steps} days", flush=True)
    arima_pred, d = forecast_arima(df["Close"].iloc[:train_end], steps=steps)
    print(f"[{label}] Random Forest forecast", flush=True)
    rf_pred = train_rf_predict(df, train_end, predict_start, predict_end, config)
    print(f"[{label}] LSTM forecast ({config.epochs} epochs)", flush=True)
    lstm_pred = fit_predict_lstm(
        df=df,
        train_end=train_end,
        predict_start=predict_start,
        predict_end=predict_end,
        look_back=config.look_back,
        units=config.lstm_units,
        epochs=config.epochs,
        batch_size=config.batch_size,
        seed=config.random_state,
    )

    predictions = pd.DataFrame(
        {
            "ARIMA": np.asarray(arima_pred, dtype=float),
            "RandomForest": np.asarray(rf_pred, dtype=float),
            "LSTM": np.asarray(lstm_pred, dtype=float),
            "NaivePreviousClose": df["Close"].shift(1).iloc[predict_start:predict_end].to_numpy(dtype=float),
        },
        index=df.index[predict_start:predict_end],
    )
    return predictions, d


def run_ticker(
    ticker: str,
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    n_rows = len(df)

    # ------------------------------------------------------------
    # DATA SPLITS
    #
    # 0%  -> 60% : Base-model training
    # 60% -> 80% : Meta-learner training (validation region)
    # 80% -> 100%: Final unseen test evaluation
    # ------------------------------------------------------------

    base_train_end = int(n_rows * 0.6)
    meta_train_end = int(n_rows * 0.8)

    # ============================================================
    # STEP 1:
    # Generate validation predictions for meta-learner training
    # ============================================================

    print(f"[{ticker}] generating meta-training predictions", flush=True)

    meta_x, meta_d = base_predictions(
        df=df,
        train_end=base_train_end,
        predict_start=base_train_end,
        predict_end=meta_train_end,
        config=config,
        label=f"{ticker} meta",
    )

    meta_y = df["Close"].iloc[base_train_end:meta_train_end].to_numpy(dtype=float)

    # ============================================================
    # STEP 2:
    # Train meta-learner ONLY on validation region
    # ============================================================

    print(f"[{ticker}] fitting leakage-free meta-learner", flush=True)

    meta_model = LinearRegression()

    meta_model.fit(
        meta_x[BASE_MODEL_COLUMNS],
        meta_y,
    )

    # ============================================================
    # STEP 3:
    # Generate FINAL TEST predictions
    # ============================================================

    print(f"[{ticker}] generating final test predictions", flush=True)

    test_x, test_d = base_predictions(
        df=df,
        train_end=meta_train_end,
        predict_start=meta_train_end,
        predict_end=n_rows,
        config=config,
        label=f"{ticker} test",
    )

    y_true = df["Close"].iloc[meta_train_end:n_rows].to_numpy(dtype=float)

    # ============================================================
    # STEP 4:
    # Hybrid prediction on UNSEEN test data
    # ============================================================

    hybrid_pred = meta_model.predict(
        test_x[BASE_MODEL_COLUMNS]
    )

    # ============================================================
    # STEP 5:
    # Collect predictions
    # ============================================================

    predictions = test_x.copy()

    predictions["Hybrid"] = hybrid_pred
    predictions["Actual"] = y_true

    # ============================================================
    # STEP 6:
    # Metrics
    # ============================================================

    rows: list[dict[str, float | str | int]] = []

    model_names = [
        *BASE_MODELS,
        "NaivePreviousClose",
        "Hybrid",
    ]

    for model_name in model_names:

        row = {
            "Ticker": ticker,
            "Model": model_name,
            **regression_metrics(
                y_true,
                predictions[model_name].to_numpy(),
            ),
        }

        rows.append(row)

    metrics = pd.DataFrame(rows)

    # ============================================================
    # Metadata
    # ============================================================

    metrics["ARIMA_d_meta"] = meta_d
    metrics["ARIMA_d_test"] = test_d

    metrics["base_train_fraction"] = 0.6
    metrics["meta_train_fraction"] = 0.2
    metrics["test_fraction"] = 0.2

    for key, value in asdict(config).items():
        metrics[key] = value

    return metrics, predictions
