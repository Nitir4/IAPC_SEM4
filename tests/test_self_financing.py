import numpy as np
import pandas as pd
import pytest

from paper_reproduction.backtesting import run_signal_backtest
from paper_reproduction.backtesting_persistent import run_signal_backtest_persistent
from paper_reproduction.signals import generate_signals


@pytest.mark.parametrize('runner', [run_signal_backtest, run_signal_backtest_persistent])
@pytest.mark.parametrize('execution', ['next_open', 'prior_close'])
def test_accounting_matches_independent_cash_share_ledger(runner, execution):
    rng = np.random.default_rng(505)
    index = pd.date_range('2020-01-01', periods=101, freq='B')
    close = 100*np.exp(np.cumsum(rng.normal(0, .02, 101)))
    market = pd.DataFrame({'Close': close, 'Open': close*np.exp(rng.normal(0, .01, 101))}, index=index)
    nums = rng.choice([-1, 0, 1], 100)
    signals = pd.DataFrame({'signal_num': nums, 'Signal': pd.Series(nums).map({-1: 'SELL', 0: 'HOLD', 1: 'BUY'}).to_numpy()}, index=index[1:])
    cost = .035
    metrics, curve = runner(market, signals, starting_cash=1000., commission=.03, slippage=.005, execution_mode=execution)
    cash, shares, state = 1000., 0., 0
    expected = []
    for i, (date, signal) in enumerate(zip(index[1:], nums), start=1):
        desired = int(signal == 1) if runner is run_signal_backtest else (1 if signal == 1 else 0 if signal == -1 else state)
        price = market.Open.loc[date] if execution == 'next_open' else market.Close.iloc[i-1]
        if desired == 1 and state == 0:
            shares = cash/(price*(1+cost))
            cash = 0.
        elif desired == 0 and state == 1:
            cash = shares*price*(1-cost)
            shares = 0.
        state = desired
        expected.append((cash+shares*market.Close.loc[date], cash, shares))
    np.testing.assert_allclose(curve[['Equity', 'Cash', 'Shares']], expected, rtol=1e-12, atol=1e-10)
    first_price = market.Open.iloc[1] if execution == 'next_open' else market.Close.iloc[0]
    assert curve.BuyHoldEquity.iloc[-1] == pytest.approx(1000*close[-1]/first_price/(1+cost))
    np.testing.assert_allclose(curve.Equity, curve.Cash+curve.Shares*curve.Close, atol=1e-10)
    assert metrics['Alpha %'] == metrics['Excess Return vs Buy Hold %']


@pytest.mark.parametrize('runner', [run_signal_backtest, run_signal_backtest_persistent])
def test_always_long_strategy_equals_net_buy_and_hold(runner):
    index = pd.date_range('2020-01-01', periods=4)
    market = pd.DataFrame({'Close': [100., 90., 120., 110.], 'Open': [100., 95., 100., 115.]}, index=index)
    signals = pd.DataFrame({'signal_num': [1, 1, 1], 'Signal': ['BUY']*3}, index=index[1:])
    metrics, curve = runner(market, signals, 100., .1, .02)
    np.testing.assert_allclose(curve.Equity, curve.BuyHoldEquity, atol=1e-13)
    assert metrics['Excess Return vs Buy Hold %'] == pytest.approx(0)
    assert metrics['Gross Buy Hold Return %'] > metrics['Buy Hold Return %']


@pytest.mark.parametrize('runner', [run_signal_backtest, run_signal_backtest_persistent])
@pytest.mark.parametrize('bad', [2, np.nan, np.inf])
def test_invalid_signal_is_not_silently_treated_as_hold(runner, bad):
    index = pd.date_range('2020-01-01', periods=3)
    market = pd.DataFrame({'Close': 100., 'Open': 100.}, index=index)
    signals = pd.DataFrame({'signal_num': [1, bad], 'Signal': ['BUY', 'HOLD']}, index=index[1:])
    with pytest.raises(ValueError, match='Signals'): runner(market, signals)


@pytest.mark.parametrize('target_mode', ['log_return', 'pct_change', 'price_change'])
def test_signal_and_risk_filters_are_invariant_to_price_currency_scale(target_mode):
    index = pd.date_range('2020-01-01', periods=4)
    market = pd.DataFrame({'Close': [100., 102., 104., 106.], 'ATR': [3., 3., 3., 3.],
                           'Volatility': .017, 'Regime': 'Bullish'}, index=index)
    forecasts = pd.DataFrame({m: [.001]*3 for m in ['ARIMA', 'RandomForest', 'LSTM']}, index=index[1:])
    if target_mode == 'price_change': forecasts[:] = .001*market.Close.iloc[:-1].to_numpy()[:, None]
    forecasts.attrs['target_mode'] = target_mode
    original = generate_signals(forecasts, market, ensemble_column='ARIMA')
    scaled_market = market.copy()
    scaled_market[['Close', 'ATR']] *= 1000.
    scaled_forecasts = forecasts.copy()
    if target_mode == 'price_change': scaled_forecasts *= 1000.
    scaled = generate_signals(scaled_forecasts, scaled_market, ensemble_column='ARIMA')
    pd.testing.assert_frame_equal(original[['Signal', 'Risk', 'Confidence']], scaled[['Signal', 'Risk', 'Confidence']])
    np.testing.assert_allclose(original.ATRFraction, scaled.ATRFraction)


def test_missing_strategy_and_agreement_model_fail_explicitly():
    index = pd.date_range('2020-01-01', periods=2)
    forecasts = pd.DataFrame({'ARIMA': [.01]}, index=index[1:])
    market = pd.DataFrame({'Close': 100., 'ATR': 1., 'Volatility': .001, 'Regime': 'Sideways'}, index=index)
    with pytest.raises(ValueError, match='strategy'): generate_signals(forecasts, market, ensemble_column='Hybrid')
    with pytest.raises(ValueError, match='agreement model'): generate_signals(forecasts, market, ensemble_column='ARIMA', agreement_columns=['missing'])
