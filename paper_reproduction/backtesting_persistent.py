from __future__ import annotations

import numpy as np
import pandas as pd


from .backtesting import _run_position_backtest


def calculate_persistent_position(signal_nums: pd.Series) -> pd.Series:
    """
    Implements IAPC_Sem4 style persistent positions.
    1 (BUY)   -> Enter/Stay Long
    -1 (SELL) -> Exit
    0 (HOLD)  -> Maintain previous state (No Action)
    """
    if not signal_nums.isin([-1, 0, 1]).all():
        raise ValueError('Signals must be -1, 0 or 1.')
    positions = np.zeros(len(signal_nums))
    current_pos = 0.0
    for i, sig in enumerate(signal_nums):
        if sig == 1:
            current_pos = 1.0
        elif sig == -1:
            current_pos = 0.0
        # If sig == 0, current_pos stays what it was
        positions[i] = current_pos
    return pd.Series(positions, index=signal_nums.index)


def run_signal_backtest_persistent(
    ohlcv_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    starting_cash: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    execution_mode: str = "next_open",
) -> tuple[dict[str, float], pd.DataFrame]:
    position = calculate_persistent_position(signals_df['signal_num'])
    return _run_position_backtest(ohlcv_df, signals_df, position, starting_cash, commission, slippage, execution_mode)


def run_strategy_backtests_persistent(
    ohlcv_df: pd.DataFrame,
    signal_frames: dict[str, pd.DataFrame],
    starting_cash: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    execution_mode: str = "next_open",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    curve_frames = []

    for strategy, signals_df in signal_frames.items():
        metrics, curve = run_signal_backtest_persistent(
            ohlcv_df=ohlcv_df,
            signals_df=signals_df,
            starting_cash=starting_cash,
            commission=commission,
            slippage=slippage,
            execution_mode=execution_mode,
        )
        metric_rows.append({"Strategy": strategy, **metrics})
        curve = curve.copy()
        curve.insert(0, "Strategy", strategy)
        curve_frames.append(curve)

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        "Final Value",
        ascending=False,
        ignore_index=True,
    )
    curves_df = pd.concat(curve_frames).rename_axis("Date").reset_index()
    return metrics_df, curves_df
