# backtesting/bt_runner.py

import backtrader as bt
import backtrader.feeds as btfeeds
import pandas as pd
import numpy as np

from backtesting.bt_strategy import MLSignalStrategy
from backtesting.bt_metrics import extract_metrics
from config import STARTING_CASH, COMMISSION


class MLDataFeed(btfeeds.PandasData):
    """
    Custom Backtrader feed with ML signal columns.
    """

    lines = (
        'signal',
        'confidence',
        'risk',
    )

    params = (
        ('datetime', None),

        # Standard OHLCV
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', -1),

        # Custom ML columns
        ('signal', 'signal_num'),
        ('confidence', 'conf_num'),
        ('risk', 'risk_num'),
    )


def encode_signals(signals_df):
    """
    Convert categorical signals to numeric values.

    BUY  ->  1
    SELL -> -1
    HOLD ->  0

    HIGH   -> 3
    MEDIUM -> 2
    LOW    -> 1
    """

    sig_map = {
        'BUY': 1,
        'SELL': -1,
        'HOLD': 0
    }

    conf_map = {
        'HIGH': 3,
        'MEDIUM': 2,
        'LOW': 1
    }

    risk_map = {
        'HIGH': 3,
        'MEDIUM': 2,
        'LOW': 1
    }

    df = signals_df.copy()

    df['signal_num'] = (
        df['Signal']
        .map(sig_map)
        .fillna(0)
        .astype(int)
    )

    df['conf_num'] = (
        df['Confidence']
        .map(conf_map)
        .fillna(1)
        .astype(int)
    )

    df['risk_num'] = (
        df['Risk']
        .map(risk_map)
        .fillna(1)
        .astype(int)
    )

    return df


def prepare_backtest_dataframe(ohlcv_df, signals_df):
    """
    Merge OHLCV + encoded signals into one dataframe
    for Backtrader.
    """

    encoded = encode_signals(signals_df)

    # Copy OHLCV
    merged = ohlcv_df.copy()

    # Flatten yfinance MultiIndex columns
    if isinstance(merged.columns, pd.MultiIndex):
        merged.columns = merged.columns.get_level_values(0)

    # Ensure datetime index
    merged.index = pd.to_datetime(merged.index)

    # Merge encoded signal columns
    merged = merged.join(
        encoded[['signal_num', 'conf_num', 'risk_num']],
        how='left'
    )

    # Fill missing values
    merged[['signal_num', 'conf_num', 'risk_num']] = (
        merged[['signal_num', 'conf_num', 'risk_num']]
        .fillna(0)
    )

    # Keep only required columns
    merged = merged[
        [
            'Open',
            'High',
            'Low',
            'Close',
            'Volume',
            'signal_num',
            'conf_num',
            'risk_num'
        ]
    ].copy()

    # Remove remaining NaNs
    merged.dropna(inplace=True)

    return merged


def run_backtest(
    ohlcv_df,
    signals_df,
    starting_cash=STARTING_CASH,
    commission=COMMISSION,
    printlog=False
):
    """
    Run Backtrader backtest.
    """

    # Prepare merged dataframe
    merged_df = prepare_backtest_dataframe(
        ohlcv_df,
        signals_df
    )

    # Create feed
    data_feed = MLDataFeed(
        dataname=merged_df
    )

    # Cerebro engine
    cerebro = bt.Cerebro()

    # Strategy
    cerebro.addstrategy(
        MLSignalStrategy,
        printlog=printlog
    )

    # Data
    cerebro.adddata(data_feed)

    # Broker settings
    cerebro.broker.setcash(starting_cash)

    cerebro.broker.setcommission(
        commission=commission
    )

    # Simulate realistic market slippage
    cerebro.broker.set_slippage_perc(0.0005)

    # Analyzers
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name='sharpe',
        riskfreerate=0.06,
        annualize=True,
        timeframe=bt.TimeFrame.Days
    )

    cerebro.addanalyzer(
        bt.analyzers.DrawDown,
        _name='drawdown'
    )

    cerebro.addanalyzer(
        bt.analyzers.TradeAnalyzer,
        _name='trades'
    )

    cerebro.addanalyzer(
        bt.analyzers.Returns,
        _name='returns'
    )

    cerebro.addanalyzer(
        bt.analyzers.TimeReturn,
        _name='timereturn'
    )

    print(f'\nStarting Portfolio: ₹{starting_cash:,.2f}')

    results = cerebro.run()

    strat = results[0]

    final_value = cerebro.broker.getvalue()

    print(f'Final Portfolio:    ₹{final_value:,.2f}')

    metrics = extract_metrics(
        strat,
        starting_cash,
        final_value
    )

    return metrics, cerebro, results


def compute_buy_and_hold(
    ohlcv_df,
    starting_cash=100000
):
    """
    Compare strategy vs buy-and-hold.
    """

    close = ohlcv_df['Close'].squeeze()

    first = float(close.iloc[0])
    last = float(close.iloc[-1])

    bh_return = round(
        ((last - first) / first) * 100,
        4
    )

    bh_value = round(
        starting_cash * (1 + (last - first) / first),
        2
    )

    print(
        f"\nBuy & Hold Return: {bh_return}%"
        f" | Final Value: ₹{bh_value:,.2f}"
    )

    return bh_return