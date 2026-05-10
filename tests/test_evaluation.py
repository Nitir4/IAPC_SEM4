import numpy as np
import pandas as pd
import pytest
from paper_reproduction.backtesting import run_signal_backtest
from paper_reproduction.backtesting_persistent import run_signal_backtest_persistent
from paper_reproduction.metrics import regression_metrics,directional_accuracy
from paper_reproduction.signals import generate_signals
from paper_reproduction.config import ExperimentConfig


def test_return_metrics_include_zero_and_lagged_benchmarks():
    actual=np.array([.1,-.2,.1]);pred=np.array([.05,-.1,0.]);lag=np.array([0.,.1,-.2])
    m=regression_metrics(actual,pred,benchmark=lag)
    assert m['R2ZeroBenchmark']==pytest.approx(1-np.sum((actual-pred)**2)/np.sum(actual**2))
    assert m['MSESkillLag1']==pytest.approx(1-np.sum((actual-pred)**2)/np.sum((actual-lag)**2))
    assert regression_metrics(actual,np.zeros(3))['R2ZeroBenchmark']==0
    assert directional_accuracy(actual,np.zeros(3))==0 # zero is an abstention, not always UP
    assert np.isnan(regression_metrics(np.zeros(3),np.zeros(3))['MAPE'])
    with pytest.raises(ValueError):regression_metrics(actual,[np.nan]*3)


@pytest.mark.parametrize('runner',[run_signal_backtest,run_signal_backtest_persistent])
def test_next_open_excludes_unavailable_overnight_gain_on_entry(runner):
    index=pd.date_range('2020-01-01',periods=4)
    prices=pd.DataFrame({'Close':[100.,120.,126.,140.],'Open':[100.,110.,120.,130.]},index=index)
    signals=pd.DataFrame({'signal_num':[1,1,-1],'Signal':['BUY','BUY','SELL']},index=index[1:])
    metrics,curve=runner(prices,signals,starting_cash=100,commission=0,slippage=0)
    # Enter at 110, stay long, exit at 130. No 100->110 overnight windfall.
    assert metrics['Final Value']==pytest.approx(100*130/110)
    assert curve['Equity'].iloc[0]==pytest.approx(100*120/110)
    assert list(curve.ExecutionClose)==[110.,120.,130.]
    assert metrics['Won']==1
    assert metrics['Execution Mode']=='next_open'
    changed=prices.copy();changed.iloc[-1,changed.columns.get_loc('Close')]=1000
    _,modified=runner(changed,signals,starting_cash=100,commission=0,slippage=0)
    pd.testing.assert_series_equal(curve.Equity,modified.Equity) # exited at the open


def test_next_open_persistent_hold_and_costs_match_roundtrip():
    index=pd.date_range('2020-01-01',periods=4)
    prices=pd.DataFrame({'Close':[100.,120.,126.,140.],'Open':[100.,110.,120.,130.]},index=index)
    signals=pd.DataFrame({'signal_num':[1,0,-1],'Signal':['BUY','HOLD','SELL']},index=index[1:])
    m,c=run_signal_backtest_persistent(prices,signals,100,.01,0)
    assert m['Final Value']==pytest.approx(100*130/110*.99/1.01)
    with pytest.raises(ValueError,match='Open'):run_signal_backtest(prices.drop(columns='Open'),signals)


@pytest.mark.parametrize('target_mode,forecast',[('log_return',np.log1p(.01)),('pct_change',.01),('price_change',2.)])
def test_signals_convert_forecast_targets_to_consistent_return_units(target_mode,forecast):
    index=pd.date_range('2020-01-01',periods=2)
    frame=pd.DataFrame({'Close':[200.,500.],'Regime':['Sideways','Bearish'],'Volatility':[.001,.1],'ATR':[1.,100.]},index=index)
    predictions=pd.DataFrame({'ARIMA':[forecast]},index=index[1:]);predictions.attrs['target_mode']=target_mode
    signal=generate_signals(predictions,frame,ensemble_column='ARIMA',buy_threshold=.009)
    assert signal.ensemble.iloc[0]==pytest.approx(.01)
    assert signal.Signal.iloc[0]=='BUY'
    assert signal.SignalReturnUnits.iloc[0]=='simple_return'


def test_direction_classification_is_not_silently_backtested_as_a_return():
    with pytest.raises(ValueError,match='classification'):ExperimentConfig(target_mode='direction')
