from __future__ import annotations

import numpy as np
import pandas as pd


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min() * 100)


def _sharpe_ratio(returns: pd.Series) -> float:
    if returns.dropna().shape[0] < 3:
        return 0.0
    returns = returns.fillna(0.0)
    return float((returns.mean() / (returns.std(ddof=0) + 1e-12)) * np.sqrt(252))


def calculate_persistent_position(signal_nums: pd.Series) -> pd.Series:
    """
    Implements IAPC_Sem4 style persistent positions.
    1 (BUY)   -> Enter/Stay Long
    -1 (SELL) -> Exit
    0 (HOLD)  -> Maintain previous state (No Action)
    """
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
) -> tuple[dict[str, float], pd.DataFrame]:
    aligned = ohlcv_df.reindex(signals_df.index).copy()
    close = aligned["Close"].astype(float)
    close_return = close.pct_change().fillna(0.0)

    # Use the persistent position logic
    position = calculate_persistent_position(signals_df["signal_num"])
    
    # Turnover only happens when the position actually changes
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (commission + slippage)
    strategy_return = position * close_return - cost

    equity = starting_cash * (1.0 + strategy_return).cumprod()
    buy_hold_equity = starting_cash * (1.0 + close_return).cumprod()

    # Metrics calculation
    trades = int((turnover > 0).sum())
    entry_dates = position[(position.diff().fillna(position) > 0)].index
    exit_dates = position[(position.diff().fillna(0) < 0)].index
    completed_returns = []
    for entry, exit_ in zip(entry_dates, exit_dates):
        entry_price = close.loc[entry]
        exit_price = close.loc[exit_]
        completed_returns.append(float(exit_price / entry_price - 1.0))

    wins = sum(ret > 0 for ret in completed_returns)
    losses = sum(ret <= 0 for ret in completed_returns)
    total_return = float(equity.iloc[-1] / starting_cash - 1.0)
    buy_hold_return = float(buy_hold_equity.iloc[-1] / starting_cash - 1.0)

    metrics = {
        "Starting Cash": starting_cash,
        "Final Value": float(equity.iloc[-1]),
        "Profit Loss": float(equity.iloc[-1] - starting_cash),
        "Total Return %": total_return * 100,
        "Buy Hold Return %": buy_hold_return * 100,
        "Alpha %": (total_return - buy_hold_return) * 100,
        "Sharpe Ratio": _sharpe_ratio(strategy_return),
        "Max Drawdown %": _max_drawdown(equity),
        "Invested Days": int((position > 0).sum()),
        "Exposure %": float((position > 0).mean() * 100),
        "Total Trades": trades,
        "Completed Trades": len(completed_returns),
        "Won": wins,
        "Lost": losses,
        "Win Rate %": (wins / len(completed_returns) * 100) if completed_returns else 0.0,
    }

    curve = pd.DataFrame(
        {
            "Close": close,
            "Signal": signals_df["Signal"],
            "Position": position,
            "MarketReturn": close_return,
            "StrategyReturn": strategy_return,
            "Equity": equity,
            "BuyHoldEquity": buy_hold_equity,
        },
        index=signals_df.index,
    )
    return metrics, curve


def run_strategy_backtests_persistent(
    ohlcv_df: pd.DataFrame,
    signal_frames: dict[str, pd.DataFrame],
    starting_cash: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
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
