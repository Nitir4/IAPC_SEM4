# plots/backtest_plots.py

import os

import pandas as pd
import matplotlib.pyplot as plt


def build_equity_dataframe(strategy, starting_cash):
    """
    Convert Backtrader strategy equity history
    into a pandas DataFrame.
    """

    equity_df = pd.DataFrame({
        'Date': strategy.date_history,
        'PortfolioValue': strategy.value_history
    })

    equity_df['Date'] = pd.to_datetime(
        equity_df['Date']
    )

    equity_df.set_index(
        'Date',
        inplace=True
    )

    # ==========================================
    # Strategy cumulative return
    # ==========================================

    equity_df['StrategyReturn'] = (
        equity_df['PortfolioValue']
        / starting_cash
    )

    return equity_df


def build_buy_hold_curve(
    close_prices,
    equity_df
):
    """
    Build benchmark aligned to strategy dates.
    """

    close_prices = close_prices.squeeze()

    # ==========================================
    # Normalize both indices to tz-naive
    # ==========================================

    close_idx = pd.to_datetime(
        close_prices.index
    )

    if close_idx.tz is not None:
        close_idx = close_idx.tz_localize(None)

    close_prices.index = close_idx

    strategy_idx = pd.to_datetime(
        equity_df.index
    )

    if strategy_idx.tz is not None:
        strategy_idx = strategy_idx.tz_localize(None)

    start_date = strategy_idx.min()
    end_date = strategy_idx.max()

    # ==========================================
    # Slice benchmark to EXACT strategy window
    # ==========================================

    aligned_close = close_prices[
        (close_prices.index >= start_date)
        & (close_prices.index <= end_date)
    ]

    # ==========================================
    # Normalize from first value in window
    # ==========================================

    normalized = (
        aligned_close / aligned_close.iloc[0]
    )

    benchmark_df = pd.DataFrame({
        'BenchmarkReturn': normalized
    })

    return benchmark_df


def plot_equity_curve(
    equity_df,
    benchmark_df,
    save_dir='outputs/plots'
):
    """
    Plot strategy vs buy-and-hold.
    """

    os.makedirs(save_dir, exist_ok=True)

    # ==========================================
    # Align benchmark to strategy dates
    # ==========================================

    benchmark_df = benchmark_df.loc[
        equity_df.index.min():
    ]

    # ==========================================
    # Plot
    # ==========================================

    plt.figure(figsize=(14, 6))

    plt.plot(
        equity_df.index,
        equity_df['StrategyReturn'],
        label='ML Strategy'
    )

    plt.plot(
        benchmark_df.index,
        benchmark_df['BenchmarkReturn'],
        label='Buy & Hold'
    )

    plt.title(
        'Strategy vs Buy-and-Hold'
    )

    plt.xlabel('Date')

    plt.ylabel('Normalized Return')

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    path = os.path.join(
        save_dir,
        'equity_curve.png'
    )

    plt.savefig(path)

    plt.close()

    print(f'Saved equity curve -> {path}')


def compute_drawdown(equity_series):
    """
    Compute drawdown series.
    """

    rolling_max = equity_series.cummax()

    drawdown = (
        equity_series - rolling_max
    ) / rolling_max

    return drawdown


def plot_drawdown(
    equity_df,
    save_dir='outputs/plots'
):
    """
    Plot drawdown chart.
    """

    os.makedirs(save_dir, exist_ok=True)

    drawdown = compute_drawdown(
        equity_df['PortfolioValue']
    )

    plt.figure(figsize=(14, 5))

    plt.plot(
        equity_df.index,
        drawdown
    )

    plt.title('Strategy Drawdown')

    plt.xlabel('Date')

    plt.ylabel('Drawdown')

    plt.grid(True)

    plt.tight_layout()

    path = os.path.join(
        save_dir,
        'drawdown.png'
    )

    plt.savefig(path)

    plt.close()

    print(f'Saved drawdown plot -> {path}')