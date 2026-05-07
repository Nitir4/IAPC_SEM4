
from __future__ import annotations

import numpy as np
import pandas as pd


def _max_drawdown(equity: pd.Series, initial_equity: float | None = None) -> float:
    running_max = equity.cummax()
    if initial_equity is not None:
        running_max = running_max.clip(lower=initial_equity)
    drawdown = equity / running_max - 1.0
    return float(drawdown.min() * 100)


def _sharpe_ratio(returns: pd.Series) -> float:
    if returns.dropna().shape[0] < 3:
        return 0.0
    returns = returns.fillna(0.0)
    return float((returns.mean() / (returns.std(ddof=0) + 1e-12)) * np.sqrt(252))


def _run_position_backtest(
    ohlcv_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    position: pd.Series,
    starting_cash: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    execution_mode: str = "next_open",
) -> tuple[dict[str, float], pd.DataFrame]:
    if signals_df.empty or not signals_df.index.is_unique or not signals_df.index.is_monotonic_increasing:
        raise ValueError('Backtest signals must be nonempty, unique and chronological.')
    if not ohlcv_df.index.is_unique or not ohlcv_df.index.is_monotonic_increasing:
        raise ValueError('Market observations must be unique and chronological.')
    if not np.isfinite([starting_cash, commission, slippage]).all() or starting_cash <= 0 or commission < 0 or slippage < 0 or commission+slippage >= 1:
        raise ValueError('Use positive starting cash and costs in [0, 1).')
    locations = ohlcv_df.index.get_indexer(signals_df.index)
    if (locations < 1).any() or (np.diff(locations) != 1).any():
        raise ValueError('Forecast dates must be consecutive market observations with a previous close.')
    if not position.index.equals(signals_df.index) or not position.isin([0, 1]).all():
        raise ValueError('Positions must align with forecast dates and be either zero or one.')
    full_close = ohlcv_df['Close'].astype(float)
    close = full_close.reindex(signals_df.index)
    previous_close = full_close.shift(1).reindex(signals_df.index)
    if execution_mode == 'next_open':
        if 'Open' not in ohlcv_df:
            raise ValueError('Next-open execution requires observed Open prices.')
        execution_close = ohlcv_df['Open'].astype(float).reindex(signals_df.index)
    elif execution_mode == 'prior_close':
        execution_close = previous_close
    else:
        raise ValueError('Execution mode must be next_open or prior_close.')
    if not np.isfinite(full_close).all() or close.isna().any() or not np.isfinite(execution_close).all() or (execution_close<=0).any() or (full_close<=0).any():
        raise ValueError('Each forecast date needs a positive current and previous observed close.')
    # Forecast row t uses information through t-1. Default execution waits
    # until Open[t]; existing positions earn the overnight move before exit.
    close_return = full_close.pct_change(fill_method=None).reindex(signals_df.index)
    if not np.isfinite(close_return).all():
        raise ValueError('Invalid returns in the backtest window.')
    prior_position = position.shift(1).fillna(0.)
    turnover = (position-prior_position).abs()
    # All-in long/cash ledger: buy q=wealth/[price*(1+c)], sell for
    # q*price*(1-c). Costs are a combined proportional execution charge.
    c = commission+slippage
    entry = position > prior_position
    exit_ = position < prior_position
    cost_factor = pd.Series(np.where(entry, 1./(1.+c), np.where(exit_, 1.-c, 1.)), index=position.index)
    if execution_mode == 'next_open':
        overnight_return = execution_close/previous_close-1.
        intraday_return = close/execution_close-1.
        gross_growth = (1.+prior_position*overnight_return)*(1.+position*intraday_return)
        # Buy-and-hold enters at the same first available open as the strategy.
        buy_hold_growth = 1.+close_return
        buy_hold_growth.iloc[0] = close.iloc[0]/execution_close.iloc[0]
    else:
        gross_growth = 1.+position*close_return
        buy_hold_growth = 1.+close_return
    strategy_return = gross_growth*cost_factor-1.

    equity = starting_cash * (1.0 + strategy_return).cumprod()
    gross_buy_hold_equity = starting_cash * buy_hold_growth.cumprod()
    # Same entry charge and terminal mark-to-market convention as the strategy.
    # Neither open position is forcibly liquidated at the final close.
    buy_hold_equity = gross_buy_hold_equity/(1.+c)

    trades = int((turnover > 0).sum())
    entry_dates = position[(position.diff().fillna(position) > 0)].index
    exit_dates = position[(position.diff().fillna(0) < 0)].index
    completed_returns = []
    for entry, exit_ in zip(entry_dates, exit_dates):
        entry_price = execution_close.loc[entry]
        exit_price = execution_close.loc[exit_]
        completed_returns.append(float(exit_price / entry_price * (1.-c)/(1.+c) - 1.0))

    wins = sum(ret > 0 for ret in completed_returns)
    losses = sum(ret <= 0 for ret in completed_returns)
    total_return = float(equity.iloc[-1] / starting_cash - 1.0)
    buy_hold_return = float(buy_hold_equity.iloc[-1] / starting_cash - 1.0)

    metrics = {
        "Starting Cash": starting_cash,
        "Execution Mode": execution_mode,
        "Final Value": float(equity.iloc[-1]),
        "Profit Loss": float(equity.iloc[-1] - starting_cash),
        "Total Return %": total_return * 100,
        "Buy Hold Return %": buy_hold_return * 100,
        "Gross Buy Hold Return %": float(gross_buy_hold_equity.iloc[-1]/starting_cash-1.)*100,
        "Excess Return vs Buy Hold %": (total_return - buy_hold_return) * 100,
        # Compatibility alias: this is benchmark excess return, not regression alpha.
        "Alpha %": (total_return - buy_hold_return) * 100,
        "Cost Convention": "all-in proportional entry/exit; terminal mark-to-market",
        "Sharpe Ratio": _sharpe_ratio(strategy_return),
        "Max Drawdown %": _max_drawdown(equity, starting_cash),
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
            "ExecutionClose": execution_close,
            "ExecutionPrice": execution_close,
            "Signal": signals_df["Signal"],
            "Position": position,
            "MarketReturn": close_return,
            "StrategyReturn": strategy_return,
            "Equity": equity,
            "BuyHoldEquity": buy_hold_equity,
            "GrossBuyHoldEquity": gross_buy_hold_equity,
            "CostFactor": cost_factor,
            "Shares": position*equity/close,
            "Cash": (1.-position)*equity,
        },
        index=signals_df.index,
    )
    return metrics, curve


def run_signal_backtest(
    ohlcv_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    starting_cash: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    execution_mode: str = "next_open",
) -> tuple[dict[str, float], pd.DataFrame]:
    if not signals_df['signal_num'].isin([-1, 0, 1]).all():
        raise ValueError('Signals must be -1, 0 or 1.')
    position = signals_df['signal_num'].replace({-1:0}).astype(float)
    return _run_position_backtest(ohlcv_df, signals_df, position, starting_cash, commission, slippage, execution_mode)


def run_strategy_backtests(
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
        metrics, curve = run_signal_backtest(
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
