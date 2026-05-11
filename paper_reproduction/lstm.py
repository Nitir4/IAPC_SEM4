from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .data import lstm_feature_columns


def set_reproducible_seeds(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


def make_sequences(values: np.ndarray, target_column: int, look_back: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values: list[np.ndarray] = []
    y_values: list[float] = []
    target_positions: list[int] = []

    for target_pos in range(look_back, len(values)):
        x_values.append(values[target_pos - look_back : target_pos])
        y_values.append(values[target_pos, target_column])
        target_positions.append(target_pos)

    return np.asarray(x_values), np.asarray(y_values), np.asarray(target_positions)


def make_target_sequences(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    look_back: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values: list[np.ndarray] = []
    y_values: list[float] = []
    target_positions: list[int] = []

    for target_pos in range(look_back, len(feature_values)):
        x_values.append(feature_values[target_pos - look_back : target_pos])
        y_values.append(target_values[target_pos])
        target_positions.append(target_pos)

    return np.asarray(x_values), np.asarray(y_values), np.asarray(target_positions)


def build_lstm(input_shape: tuple[int, int], units: int):
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(units, return_sequences=True),
            LSTM(units),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(), loss="mean_squared_error")
    return model


def inverse_close(values: np.ndarray, scaler: MinMaxScaler, close_index: int, num_features: int) -> np.ndarray:
    padded = np.zeros((len(values), num_features))
    padded[:, close_index] = values.reshape(-1)
    return scaler.inverse_transform(padded)[:, close_index]


def fit_predict_lstm(
    df: pd.DataFrame,
    target: pd.Series,
    train_end: int,
    predict_start: int,
    predict_end: int,
    look_back: int,
    units: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    set_reproducible_seeds(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)

    feature_columns = lstm_feature_columns()
    values = df[feature_columns].to_numpy(dtype=float)
    target_values = target.to_numpy(dtype=float).reshape(-1, 1)

    feature_scaler = MinMaxScaler()
    feature_scaler.fit(values[:train_end])
    scaled_features = feature_scaler.transform(values)

    target_scaler = StandardScaler()
    train_target = target_values[:train_end]
    target_scaler.fit(train_target[~np.isnan(train_target).reshape(-1)])
    scaled_target = target_scaler.transform(target_values).reshape(-1)

    x_all, y_all, target_positions = make_target_sequences(scaled_features, scaled_target, look_back)
    train_mask = (target_positions < train_end) & ~np.isnan(y_all)
    predict_mask = (target_positions >= predict_start) & (target_positions < predict_end)

    model = build_lstm((look_back, len(feature_columns)), units)
    model.fit(
        x_all[train_mask],
        y_all[train_mask],
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        shuffle=False,
    )
    pred_scaled = model.predict(x_all[predict_mask], verbose=0).reshape(-1)
    return target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
