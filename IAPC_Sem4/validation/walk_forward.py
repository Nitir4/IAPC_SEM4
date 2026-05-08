# validation/walk_forward.py

import numpy as np
import pandas as pd
from data.preprocess import get_features_target, scale_features
from validation.metrics import compute_metrics
from models.trainer import MODELS

def walk_forward_validation(df, n_splits=5):
    """
    Expanding window walk-forward validation.
    Trains each model on growing history, tests on next window.
    Scaler is fit fresh each window — no leakage.
    """
    results = {name: [] for name in MODELS.keys()}

    total = len(df)
    # minimum 60% for first train window
    min_train = int(total * 0.60)
    step = int((total - min_train) / n_splits)

    print(f"\nWalk-Forward Validation ({n_splits} splits)")
    print(f"Total rows: {total}, Min train: {min_train}, Step: {step}")

    for i in range(n_splits):
        train_end  = min_train + i * step
        test_start = train_end
        test_end   = min(train_end + step, total)

        if test_end <= test_start:
            break

        train_df = df.iloc[:train_end]
        test_df  = df.iloc[test_start:test_end]

        X_train, y_train = get_features_target(train_df)
        X_test,  y_test  = get_features_target(test_df)

        # Fresh scaler per window — critical
        X_train_sc, X_test_sc, _ = scale_features(X_train, X_test)

        print(f"\n  Window {i+1}: Train→{train_df.index[-1].date()} "
              f"| Test {test_df.index[0].date()}→{test_df.index[-1].date()}")

        for name, get_model in MODELS.items():
            model = get_model()
            model.fit(X_train_sc, y_train)
            preds = model.predict(X_test_sc)
            m = compute_metrics(y_test, preds)
            m['window'] = i + 1
            results[name].append(m)

    return summarize_wf_results(results)

def summarize_wf_results(results):
    """Average metrics across all windows per model."""
    print(f"\n{'='*50}")
    print("Walk-Forward Summary (averaged across windows):")
    summary = {}
    for name, windows in results.items():
        avg = {
            'MAE': round(np.mean([w['MAE'] for w in windows]), 6),
            'RMSE': round(np.mean([w['RMSE'] for w in windows]), 6),
            'R2': round(np.mean([w['R2'] for w in windows]), 4),
            'DirectionalAccuracy': round(
                np.mean([w['DirectionalAccuracy'] for w in windows]), 4)
        }
        summary[name] = avg
        print(f"\n  {name}:")
        for k, v in avg.items():
            print(f"    {k}: {v}")
    return summary