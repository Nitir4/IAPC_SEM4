# plots/test_backtest_plots.py

import pandas as pd

from data.loader import get_data

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
# Load OHLCV
# ==========================================

ohlcv_df = get_data()

# ==========================================
# Load signals
# ==========================================

signals_df = pd.read_csv(
    'outputs/signals.csv',
    index_col=0,
    parse_dates=True
)

# ==========================================
# Run backtest
# ==========================================

metrics, cerebro, results = run_backtest(
    ohlcv_df,
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
# Build benchmark
# ==========================================

benchmark_df = build_buy_hold_curve(
    close_prices=ohlcv_df['Close'],
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