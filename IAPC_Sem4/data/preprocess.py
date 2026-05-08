# data/preprocess.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import TARGET_COLUMN, TRAIN_RATIO

def compute_log_returns(df):
    """Compute log returns from Close price."""
    df = df.copy()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    return df

def add_forward_return(df, shift=1):
    """
    Create target: next-day log return.
    This is what we are predicting.
    """
    df = df.copy()
    df['target'] = df['log_return'].shift(-shift)
    return df

def clean_data(df):
    """Remove NaNs created by indicators and shifts."""
    df = df.copy()
    df.dropna(inplace=True)
    return df

def split_data(df, train_ratio=TRAIN_RATIO):
    """
    Chronological train/test split.
    NO shuffling. Time order must be preserved.
    """
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    print(f"Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} rows)")
    print(f"Test:  {test.index[0].date()} → {test.index[-1].date()} ({len(test)} rows)")
    return train, test

def get_features_target(df, target_col='target'):
    """
    Separate features (X) from target (y).
    Drops non-feature columns.
    """
    drop_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                 'log_return', 'target', 'Regime']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df[target_col]
    return X, y

def scale_features(X_train, X_test):
    """
    Fit scaler on train only. Transform both.
    CRITICAL: never fit on full data before splitting.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler