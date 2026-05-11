# main.py

import os
import sys
import json
import argparse
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
    get_features_target
)

from data.indicators import add_indicators

from data.regime import (
    detect_regime,
    regime_summary
)

from models.trainer import (
    MODELS,
    train_all_models,
    predict_all_models
)

from models.ensembles import (
    residual_hybrid_predict,
    confidence_weighted_predict
)

from validation.metrics import compute_metrics

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
from config import OUTPUT_DIR, TICKERS

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

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
# WALK-FORWARD UTILITY
# ============================================================

def walk_forward_base_predictions(df, predict_start, predict_end, refit_every=20):
    """
    Generate predictions using a walk-forward approach.
    Re-trains base models every 'refit_every' steps.
    """
    all_preds = {name: [] for name in MODELS.keys()}
    total_steps = predict_end - predict_start
    
    print(f"  Generating walk-forward predictions for {total_steps} steps (refit every {refit_every} days)...")
    
    for i in range(predict_start, predict_end, refit_every):
        current_train_end = i
        current_predict_end = min(i + refit_every, predict_end)
        
        train_df = df.iloc[:current_train_end]
        predict_df = df.iloc[current_train_end:current_predict_end]
        
        if predict_df.empty:
            break
            
        X_train, y_train = get_features_target(train_df)
        X_predict, y_predict = get_features_target(predict_df)
        
        # Fit scaler on available history
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_predict_scaled = scaler.transform(X_predict)
        
        # Fit models on available history and predict
        for name, get_model in MODELS.items():
            model = get_model()
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_predict_scaled)
            all_preds[name].extend(preds)
            
    return {name: np.array(preds) for name, preds in all_preds.items()}

# ============================================================
# MAIN PIPELINE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Run the IAPC Sem4 stock analysis pipeline.')
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=TICKERS,
        help='Ticker CSV names to load from the repo-level data directory.',
    )
    parser.add_argument(
        '--output-dir',
        default=OUTPUT_DIR,
        help='Directory where Sem4 outputs should be written.',
    )
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Download with yfinance if set; otherwise local repo CSVs are used.',
    )
    parser.add_argument(
        '--refit-every',
        type=int,
        default=20,
        help='How many days between model re-trainings in walk-forward phase.',
    )
    return parser.parse_args()


def run_pipeline_for_ticker(ticker, output_dir, force_download=False, refit_every=20):

    print('\n==============================')
    print(f'AI STOCK ANALYSIS PIPELINE: {ticker}')
    print('==============================')

    ticker_output_dir = os.path.join(output_dir, ticker)
    ticker_plots_dir = os.path.join(ticker_output_dir, 'plots')
    os.makedirs(ticker_output_dir, exist_ok=True)
    os.makedirs(ticker_plots_dir, exist_ok=True)

    # ========================================================
    # LOAD DATA
    # ========================================================

    try:
        df = get_data(ticker=ticker, force_download=force_download)
    except Exception as e:
        print(f'\nERROR: Failed to load data: {e}')
        raise

    if df.empty:
        print('\nERROR: Loaded dataframe is empty.')
        sys.exit(1)

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
    # TRAIN / META / TEST SPLIT (60/20/20) -> TOTAL 80/20
    # ========================================================

    n = len(df)
    split_idx_1 = int(n * 0.6)
    split_idx_2 = int(n * 0.8)

    # Note: Ensembles need a 'meta' set to fit their weights.
    # We use 60% for base model initial training, 20% for ensemble weight fitting, 
    # and the final 20% for out-of-sample testing.
    # The total 'training history' for the final test is 80%.

    test_df  = df.iloc[split_idx_2:].copy()
    _, y_meta = get_features_target(df.iloc[split_idx_1:split_idx_2])
    _, y_test = get_features_target(test_df)

    print(f'\nSplit Indices: Base Train end={split_idx_1}, Meta end={split_idx_2}, Total={n}')

    # ========================================================
    # WALK-FORWARD BASE PREDICTIONS
    # ========================================================

    print(f'\nGenerating Meta Range Base Predictions...')
    meta_preds_dict = walk_forward_base_predictions(df, split_idx_1, split_idx_2, refit_every=refit_every)
    meta_preds_df = pd.DataFrame(meta_preds_dict, index=df.index[split_idx_1:split_idx_2])

    print(f'\nGenerating Test Range Base Predictions...')
    test_preds_dict = walk_forward_base_predictions(df, split_idx_2, n, refit_every=refit_every)
    test_preds_df = pd.DataFrame(test_preds_dict, index=df.index[split_idx_2:])

    # ========================================================
    # ENSEMBLES
    # ========================================================

    print('\nCalculating Ensembles...')
    
    # Residual Hybrid (using RandomForest as base if available, else first model)
    base_model = 'RandomForest' if 'RandomForest' in meta_preds_df.columns else meta_preds_df.columns[0]
    res_hybrid_pred, _ = residual_hybrid_predict(meta_preds_df, y_meta.values, test_preds_df, base_model_name=base_model)
    test_preds_df['ResidualHybrid'] = res_hybrid_pred

    # Confidence Weighted
    conf_weighted_pred = confidence_weighted_predict(test_preds_df[list(meta_preds_dict.keys())])
    test_preds_df['ConfidenceWeight'] = conf_weighted_pred

    # Add Actual
    test_preds_df['Actual'] = y_test.values

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    predictions_path = os.path.join(ticker_output_dir, 'predictions.csv')
    test_preds_df.to_csv(predictions_path)
    print(f'\nSaved predictions -> {predictions_path}')

    # ========================================================
    # CALCULATE REGRESSION METRICS & BACKTEST ALL
    # ========================================================

    all_model_metrics = {}
    backtest_results = []
    
    model_columns = [c for c in test_preds_df.columns if c != 'Actual']
    
    for model_name in model_columns:
        # Regression metrics
        m = compute_metrics(y_test.values, test_preds_df[model_name].values, label=model_name)
        all_model_metrics[model_name] = m
        
        # Signal Generation
        signals_df = generate_signals(
            predictions_dict=test_preds_df.to_dict(orient='list'),
            df_index=test_preds_df.index,
            regime_series=test_df['Regime'],
            volatility_series=test_df['Volatility'],
            atr_series=test_df['ATR'],
            ensemble_column=model_name
        )
        
        # Backtest
        try:
            bt_metrics, cerebro, results = run_backtest(
                ohlcv_df=test_df,
                signals_df=signals_df,
                printlog=False
            )
            bt_metrics['Strategy'] = model_name
            # Explicitly add Profit Loss
            bt_metrics['Profit Loss'] = bt_metrics['Final Value'] - bt_metrics['Starting Cash']
            backtest_results.append(bt_metrics)
            
            # Save equity curve for the best model or preferred one (e.g., ConfidenceWeight)
            if model_name == 'ConfidenceWeight':
                strategy = results[0]
                equity_df = build_equity_dataframe(strategy=strategy, starting_cash=100000)
                benchmark_df = build_buy_hold_curve(close_prices=test_df['Close'], equity_df=equity_df)
                plot_equity_curve(equity_df, benchmark_df, save_dir=ticker_plots_dir)
                plot_drawdown(equity_df, save_dir=ticker_plots_dir)
                equity_df.to_csv(os.path.join(ticker_output_dir, 'equity_curve.csv'))

        except Exception as e:
            print(f'WARNING: Backtest failed for {model_name}: {e}')

    # Save detailed metrics
    with open(os.path.join(ticker_output_dir, 'model_metrics.json'), 'w') as f:
        json.dump(all_model_metrics, f, indent=4)

    # Buy & Hold benchmark
    bh_return = compute_buy_and_hold(test_df)
    
    # Build results dataframes
    metrics_df = pd.DataFrame([
        {'Ticker': ticker, 'Model': k, **v} for k, v in all_model_metrics.items()
    ])
    backtests_df = pd.DataFrame(backtest_results)
    backtests_df['Ticker'] = ticker
    backtests_df['BuyHoldReturn %'] = bh_return
    backtests_df['Alpha %'] = round(backtests_df['Total Return %'] - bh_return, 4)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print('\nRegression Metrics:')
    print(metrics_df[['Ticker', 'Model', 'MAE', 'RMSE', 'R2', 'DirectionalAccuracy', 'IC']].to_string(index=False))

    print('\nBacktest Results:')
    backtest_cols = ['Ticker', 'Strategy', 'Final Value', 'Profit Loss', 'Total Return %', 'BuyHoldReturn %', 'Alpha %', 'Sharpe Ratio', 'Max Drawdown %', 'Total Trades']
    # Filter columns to only those that exist (to avoid future key errors)
    existing_cols = [c for c in backtest_cols if c in backtests_df.columns]
    print(backtests_df[existing_cols].to_string(index=False))

    print(f'\nPipeline completed successfully for {ticker}.')
    
    # Return best strategy based on Sharpe
    try:
        best_idx = backtests_df['Sharpe Ratio'].idxmax()
        return {
            'Ticker': ticker,
            **backtests_df.iloc[best_idx].to_dict(),
        }
    except:
        return {'Ticker': ticker}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for ticker in args.tickers:
        try:
            res = run_pipeline_for_ticker(
                    ticker=ticker,
                    output_dir=args.output_dir,
                    force_download=args.force_download,
                    refit_every=args.refit_every
                )
            if res:
                all_results.append(res)
        except Exception as e:
            print(f'\nERROR: Pipeline failed for {ticker}: {e}')
            continue

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(args.output_dir, 'summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f'\nSaved Sem4 summary -> {summary_path}')
    else:
        print('\nERROR: No ticker completed successfully.')
        sys.exit(1)

if __name__ == '__main__':
    main()
