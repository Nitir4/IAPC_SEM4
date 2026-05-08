# plots/test_backtest_plots.py

import pandas as pd

from data.loader import get_data

from data.preprocess import (
    compute_log_returns,
    add_forward_return,
    clean_data,
    split_data
)

from data.indicators import add_indicators

from data.regime import detect_regime

from backtesting.bt_runner import (
    run_backtest
)

from plots.backtest_plots import (
    build_equity_dataframe,
    build_buy_hold_curve,
    plot_equity_curve,
    plot_drawdown
)

# ==========================================
# Load and preprocess OHLCV
# ==========================================

ohlcv_df = get_data()

ohlcv_df = compute_log_returns(ohlcv_df)
ohlcv_df = add_indicators(ohlcv_df)
ohlcv_df = add_forward_return(ohlcv_df)
ohlcv_df = clean_data(ohlcv_df)
ohlcv_df = detect_regime(ohlcv_df)

# ==========================================
# Split into train/test
# ==========================================

_, test_df = split_data(ohlcv_df)

# ==========================================
# Load signals
# ==========================================

signals_df = pd.read_csv(
    'outputs/signals.csv',
    index_col=0,
    parse_dates=True
)

# ==========================================
# Run backtest (on test data only)
# ==========================================

metrics, cerebro, results = run_backtest(
    test_df,
    signals_df,
    printlog=False
)

strategy = results[0]

# ==========================================
# Build equity dataframe
# ==========================================

equity_df = build_equity_dataframe(
    strategy=strategy,
    starting_cash=100000
)

# ==========================================
# Build benchmark (test period only)
# ==========================================

benchmark_df = build_buy_hold_curve(
    close_prices=test_df['Close'],
    equity_df=equity_df
)

# ==========================================
# Generate plots
# ==========================================

plot_equity_curve(
    equity_df,
    benchmark_df
)

plot_drawdown(
    equity_df
)

print('\nBacktest plots generated.')