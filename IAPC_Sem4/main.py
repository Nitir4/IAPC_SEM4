# main.py

import os
import json
import numpy as np
import pandas as pd

# ============================================================
# Prevent sklearn/joblib thread explosion on Windows
# ============================================================

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

# ============================================================
# Reproducibility
# ============================================================

np.random.seed(42)

# ============================================================
# Imports
# ============================================================

from data.loader import get_data

from data.preprocess import (
    compute_log_returns,
    add_forward_return,
    clean_data,
    split_data,
    get_features_target,
    scale_features
)

from data.indicators import add_indicators

from data.regime import (
    detect_regime,
    regime_summary
)

from models.trainer import (
    train_all_models,
    predict_all_models
)

from signals.signal_generator import (
    generate_signals
)

from backtesting.bt_runner import (
    run_backtest,
    compute_buy_and_hold
)

from plots.forecast_plots import (
    plot_model_forecast
)

from plots.backtest_plots import (
    build_equity_dataframe,
    build_buy_hold_curve,
    plot_equity_curve,
    plot_drawdown
)

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = 'outputs'

PLOTS_DIR = os.path.join(
    OUTPUT_DIR,
    'plots'
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print('\n==============================')
    print('AI STOCK ANALYSIS PIPELINE')
    print('==============================')

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = get_data()

    print(f'\nRaw Shape: {df.shape}')

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    df = compute_log_returns(df)

    df = add_indicators(df)

    df = add_forward_return(df)

    df = clean_data(df)

    df = detect_regime(df)

    regime_summary(df)

    print(f'\nProcessed Shape: {df.shape}')

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    train_df, test_df = split_data(df)

    X_train, y_train = get_features_target(
        train_df
    )

    X_test, y_test = get_features_target(
        test_df
    )

    # ========================================================
    # SCALE FEATURES
    # ========================================================

    X_train_scaled, X_test_scaled, scaler = (
        scale_features(
            X_train,
            X_test
        )
    )

    print(f'\nX_train shape: {X_train_scaled.shape}')
    print(f'X_test shape:  {X_test_scaled.shape}')

    # ========================================================
    # TRAIN MODELS
    # ========================================================

    trained_models = train_all_models(
        X_train_scaled,
        y_train
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    predictions_dict = predict_all_models(
        trained_models,
        X_test_scaled
    )

    predictions_df = pd.DataFrame(
        predictions_dict,
        index=X_test.index
    )

    predictions_df['Actual'] = y_test.values

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    predictions_path = os.path.join(
        OUTPUT_DIR,
        'predictions.csv'
    )

    predictions_df.to_csv(
        predictions_path
    )

    print(
        f'\nSaved predictions -> {predictions_path}'
    )

    # ========================================================
    # FORECAST PLOTS
    # ========================================================

    for model_name in trained_models.keys():

        plot_model_forecast(
            dates=X_test.index,
            actual=y_test.values,
            predicted=predictions_dict[model_name],
            model_name=model_name
        )

    # ========================================================
    # SIGNAL GENERATION
    # ========================================================

    signals_df = generate_signals(
        predictions_dict=predictions_dict,
        df_index=X_test.index,
        regime_series=test_df['Regime'],
        volatility_series=test_df['Volatility'],
        atr_series=test_df['ATR']
    )

    # ========================================================
    # SAVE SIGNALS
    # ========================================================

    signals_path = os.path.join(
        OUTPUT_DIR,
        'signals.csv'
    )

    signals_df.to_csv(
        signals_path
    )

    print(
        f'Saved signals -> {signals_path}'
    )

    # ========================================================
    # BACKTEST
    # ========================================================

    metrics, cerebro, results = run_backtest(
        ohlcv_df=test_df,
        signals_df=signals_df,
        printlog=False
    )

    strategy = results[0]

    # ========================================================
    # BUY & HOLD
    # ========================================================

    bh_return = compute_buy_and_hold(
        test_df
    )

    metrics['BuyHoldReturn %'] = bh_return

    metrics['Alpha %'] = round(
        metrics['Total Return %'] - bh_return,
        4
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_path = os.path.join(
        OUTPUT_DIR,
        'metrics.json'
    )

    with open(
        metrics_path,
        'w'
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print(
        f'Saved metrics -> {metrics_path}'
    )

    # ========================================================
    # EQUITY CURVE
    # ========================================================

    equity_df = build_equity_dataframe(
        strategy=strategy,
        starting_cash=100000
    )

    benchmark_df = build_buy_hold_curve(
        close_prices=test_df['Close'],
        equity_df=equity_df
    )

    plot_equity_curve(
        equity_df,
        benchmark_df
    )

    plot_drawdown(
        equity_df
    )

    # ========================================================
    # SAVE EQUITY DATA
    # ========================================================

    equity_path = os.path.join(
        OUTPUT_DIR,
        'equity_curve.csv'
    )

    equity_df.to_csv(
        equity_path
    )

    print(
        f'Saved equity curve -> {equity_path}'
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print('\n==============================')
    print('FINAL RESULTS')
    print('==============================')

    for k, v in metrics.items():

        print(f'{k:<20}: {v}')

    print('\nPipeline completed successfully.')

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':

    main()