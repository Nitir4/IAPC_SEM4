# backtesting/bt_metrics.py

import numpy as np

def extract_metrics(strategy, starting_cash, final_value):
    total_return = ((final_value - starting_cash) / starting_cash) * 100

    # Drawdown
    dd_analysis = strategy.analyzers.drawdown.get_analysis()
    max_dd      = round(dd_analysis.get('max', {}).get('drawdown', 0), 4)

    # Trades
    trade_analysis = strategy.analyzers.trades.get_analysis()
    total_trades   = trade_analysis.get('total', {}).get('closed', 0)
    won            = trade_analysis.get('won',  {}).get('total', 0)
    lost           = trade_analysis.get('lost', {}).get('total', 0)
    win_rate       = round((won / total_trades * 100), 2) if total_trades > 0 else 0

    # Manual Sharpe from daily time returns
    time_return    = strategy.analyzers.timereturn.get_analysis()
    daily_returns  = list(time_return.values())

    if len(daily_returns) > 2:
        ret_arr     = np.array(daily_returns)
        excess      = ret_arr - (0.06 / 252)   # daily risk-free
        sharpe      = round((np.mean(excess) / (np.std(excess) + 1e-9)) * np.sqrt(252), 4)
    else:
        sharpe = None

    # Buy and hold return for comparison
    metrics = {
        'Starting Cash':  starting_cash,
        'Final Value':    round(final_value, 2),
        'Total Return %': round(total_return, 4),
        'Sharpe Ratio':   sharpe if sharpe is not None else 'N/A (too few returns)',
        'Max Drawdown %': max_dd,
        'Total Trades':   total_trades,
        'Won':            won,
        'Lost':           lost,
        'Win Rate %':     win_rate,
    }

    print(f"\n{'='*45}")
    print("BACKTEST RESULTS")
    print(f"{'='*45}")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")
    print(f"{'='*45}")

    return metrics