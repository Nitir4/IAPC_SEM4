from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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
    close_index = feature_columns.index("Close")

    scaler = MinMaxScaler()
    scaler.fit(values[:train_end])
    scaled = scaler.transform(values)

    x_all, y_all, target_positions = make_sequences(scaled, close_index, look_back)
    train_mask = target_positions < train_end
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
    return inverse_close(pred_scaled, scaler, close_index, len(feature_columns))
