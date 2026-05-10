import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from paper_reproduction import arima, ensemble, lstm
from paper_reproduction.config import ExperimentConfig, split_boundaries
from paper_reproduction.data import add_technical_features, build_target, load_stock_csv
from paper_reproduction.signals import generate_signals
from paper_reproduction.backtesting import run_signal_backtest
from paper_reproduction.backtesting_persistent import run_signal_backtest_persistent


def market(n=120):
    rng=np.random.default_rng(505)
    close=100*np.exp(np.cumsum(rng.normal(0,.01,n)))
    return pd.DataFrame({'Close':close,'High':close*1.01,'Low':close*.99,'Open':close*.998,'Volume':1000},
                        index=pd.date_range('2020-01-01',periods=n,freq='B',name='Date'))


def test_csv_and_features_do_not_use_future_rows(tmp_path):
    raw=market(); raw.iloc[0,0]=np.nan; raw.iloc[50,0]=np.nan
    first=tmp_path/'first.csv';raw.to_csv(first)
    original=load_stock_csv(first)
    changed=raw.copy();changed.iloc[70:,:4]*=2
    second=tmp_path/'second.csv';changed.to_csv(second)
    revised=load_stock_csv(second)
    pd.testing.assert_frame_equal(original.iloc[:69],revised.iloc[:69])
    assert original.loc[raw.index[50],'Close']==raw.loc[raw.index[49],'Close']
    assert original.index[0]==raw.index[1]
    assert original['return_lag3'].iloc[:3].isna().all()


def test_invalid_csv_dates_and_prices_are_rejected(tmp_path):
    raw=market(10);raw.index=[raw.index[0]]*10
    raw.index.name='Date'
    path=tmp_path/'duplicate.csv';raw.to_csv(path)
    with pytest.raises(ValueError,match='unique'):load_stock_csv(path)
    raw=market(10);raw.iloc[0,0]=-1;raw.to_csv(path)
    with pytest.raises(ValueError,match='positive'):load_stock_csv(path)


def test_split_configuration_and_old_alias_are_truthful():
    assert split_boundaries(100,ExperimentConfig())==(60,80)
    custom=ExperimentConfig(train_fraction=.75,meta_fraction_of_train=1/3)
    assert split_boundaries(120,custom)==(60,90)
    with pytest.raises(ValueError):split_boundaries(30,ExperimentConfig())
    with pytest.raises(ValueError):ExperimentConfig(train_fraction=1)
    with pytest.warns(UserWarning,match='obsolete'):
        assert ExperimentConfig(stacking_mode='paper_like_in_sample').stacking_mode=='chronological_holdout'


def test_regime_selection_and_signal_filters_use_previous_observation():
    frame=add_technical_features(market())
    day=frame.index[80]
    frame.loc[frame.index[79],'Regime']='Bullish'
    frame.loc[day,'Regime']='Bearish'
    predictions=pd.DataFrame({'ARIMA':[.001],'RandomForest':[.002],'LSTM':[.003]},index=[day])
    before=ensemble.regime_switching_predict(predictions,frame)
    frame.loc[day,'Regime']='Volatile'
    assert ensemble.regime_switching_predict(predictions,frame)[0]==before[0]==.003
    frame.loc[frame.index[79],['Regime','Volatility','ATR']]=['Sideways',.001,1]
    predicted=pd.DataFrame({'ARIMA':[.001],'RandomForest':[.001],'LSTM':[.001]},index=[day])
    signal=generate_signals(predicted,frame,ensemble_column='ARIMA')
    frame.loc[day,['Volatility','ATR']]=[1,100]
    changed=generate_signals(predicted,frame,ensemble_column='ARIMA')
    pd.testing.assert_frame_equal(signal,changed)
    assert signal['Signal'].iloc[0]=='BUY'


def test_random_forest_prediction_at_t_does_not_see_t_close():
    raw=market();first=add_technical_features(raw)
    changed=raw.copy();changed.iloc[75:,:4]*=2
    second=add_technical_features(changed)
    config=ExperimentConfig(rf_estimators=5)
    before=ensemble.train_rf_predict(first,60,70,80,config,'log_return')
    after=ensemble.train_rf_predict(second,60,70,80,config,'log_return')
    np.testing.assert_allclose(before[:6],after[:6])


def test_lstm_training_excludes_warmup_and_prediction_windows_are_causal(monkeypatch):
    captures=[]
    class Model:
        def fit(self,x,y,**kwargs):
            assert np.isfinite(x).all() and np.isfinite(y).all()
            captures.append(x.copy())
        def predict(self,x,**kwargs):
            captures.append(x.copy())
            return x.mean(axis=(1,2))[:,None]
    monkeypatch.setitem(sys.modules,'tensorflow',SimpleNamespace(random=SimpleNamespace(set_seed=lambda seed:None)))
    monkeypatch.setattr(lstm,'build_lstm',lambda shape,units:Model())
    raw=market();first=add_technical_features(raw)
    changed=raw.copy();changed.iloc[75:,:4]*=2
    second=add_technical_features(changed)
    kwargs=dict(train_end=60,predict_start=70,predict_end=80,look_back=10,units=4,epochs=1,batch_size=4,seed=505)
    before=lstm.fit_predict_lstm(first,build_target(first),**kwargs)
    after=lstm.fit_predict_lstm(second,build_target(second),**kwargs)
    np.testing.assert_array_equal(captures[0],captures[2])
    np.testing.assert_allclose(before[:6],after[:6])


def test_arima_walk_forward_does_not_consume_current_or_future_target():
    target=build_target(add_technical_features(market()))
    changed=target.copy();changed.iloc[73:]+=1
    first,_=arima.rolling_arima_return_forecast(target,60,70,75,refit_every=2)
    second,_=arima.rolling_arima_return_forecast(changed,60,70,75,refit_every=2)
    np.testing.assert_allclose(first.iloc[:4],second.iloc[:4])


def test_second_differencing_is_not_a_two_period_first_difference(monkeypatch):
    seen=[]
    monkeypatch.setattr(arima,'adfuller',lambda values,**kwargs:seen.append(values.copy()) or (0,1))
    arima.choose_d(pd.Series(np.arange(30,dtype=float)**2))
    assert [len(values) for values in seen]==[30,29,28]
    np.testing.assert_array_equal(seen[2],np.full(28,2.))


def test_test_labels_cannot_fit_meta_weights_and_metadata_matches_splits(monkeypatch):
    frame=add_technical_features(market())
    calls=[]
    def base(df,target,train_end,predict_start,predict_end,**kwargs):
        calls.append((train_end,predict_start,predict_end))
        n=predict_end-predict_start
        return pd.DataFrame({'ARIMA':np.linspace(-.01,.01,n),'RandomForest':np.linspace(.01,-.01,n),
            'LSTM':np.zeros(n),'NaiveZero':np.zeros(n),'NaiveLag1':np.zeros(n)},index=df.index[predict_start:predict_end]),0
    monkeypatch.setattr(ensemble,'base_predictions',base)
    config=ExperimentConfig(train_fraction=.75,meta_fraction_of_train=1/3)
    metrics,before=ensemble.run_ticker('TEST',frame,config)
    changed=frame.copy();changed.iloc[90:,changed.columns.get_loc('Close')]*=2
    _,after=ensemble.run_ticker('TEST',changed,config)
    np.testing.assert_allclose(before['ResidualHybrid'],after['ResidualHybrid'])
    assert calls[:2]==[(60,60,90),(90,90,120)]
    assert metrics['base_train_fraction'].iloc[0]==.5
    assert metrics['test_fraction'].iloc[0]==.25


@pytest.mark.parametrize('runner',[run_signal_backtest,run_signal_backtest_persistent])
def test_backtest_keeps_first_forecast_return_and_initial_drawdown(runner):
    index=pd.date_range('2020-01-01',periods=4)
    prices=pd.DataFrame({'Close':[100,110,121,121]},index=index)
    signals=pd.DataFrame({'signal_num':[1,0,-1],'Signal':['BUY','HOLD','SELL']},index=index[1:])
    metrics,curve=runner(prices,signals,starting_cash=100,commission=0,slippage=0,execution_mode='prior_close')
    assert curve['MarketReturn'].iloc[0]==pytest.approx(.1)
    assert curve['Equity'].iloc[0]==pytest.approx(110)
    assert curve['ExecutionClose'].iloc[0]==100
    prices['Close']=100
    metrics,_=runner(prices,signals,starting_cash=100,commission=.01,slippage=0,execution_mode='prior_close')
    assert metrics['Final Value']==pytest.approx(100*.99/1.01)
    assert metrics['Max Drawdown %']==pytest.approx((.99/1.01-1)*100)
    assert metrics['Won']==0 and metrics['Lost']==1


def test_persistent_hold_and_roundtrip_costs_match_execution_prices():
    index=pd.date_range('2020-01-01',periods=4)
    prices=pd.DataFrame({'Close':[100,110,121,121]},index=index)
    signals=pd.DataFrame({'signal_num':[1,0,-1],'Signal':['BUY','HOLD','SELL']},index=index[1:])
    metrics,curve=run_signal_backtest_persistent(prices,signals,100,.01,0,execution_mode='prior_close')
    assert metrics['Final Value']==pytest.approx(100*1.21*.99/1.01)
    assert list(curve['Position'])==[1,1,0]


@pytest.mark.parametrize('runner',[run_signal_backtest,run_signal_backtest_persistent])
def test_backtests_reject_gaps_and_nonfinite_parameters(runner):
    prices=market(5)
    signals=pd.DataFrame({'signal_num':[1,1],'Signal':['BUY','BUY']},index=prices.index[[1,3]])
    with pytest.raises(ValueError,match='consecutive'):
        runner(prices,signals)
    signals.index=prices.index[1:3]
    with pytest.raises(ValueError,match='positive starting cash'):
        runner(prices,signals,starting_cash=np.nan)
    prices.loc[prices.index[2],'Close']=np.inf
    with pytest.raises(ValueError,match='positive current'):
        runner(prices,signals)


@pytest.mark.parametrize('module_name',['run','run_persistent'])
def test_runners_accept_split_configuration_and_reject_inactive_options(module_name,monkeypatch):
    import importlib
    module=importlib.import_module('paper_reproduction.'+module_name)
    monkeypatch.setattr(sys,'argv',['run','--train-fraction','.75','--meta-fraction-of-train','.3333333333333333'])
    args=module.parse_args()
    assert args.train_fraction==.75 and args.stacking_mode=='chronological_holdout'
    monkeypatch.setattr(sys,'argv',['run','--meta-max-weight','.9'])
    with pytest.raises(SystemExit):module.parse_args()


def test_runner_exports_configured_splits_with_real_rf_and_arima(tmp_path,monkeypatch):
    from paper_reproduction import run
    data=tmp_path/'data';data.mkdir()
    market().to_csv(data/'TEST.csv')
    output=tmp_path/'outputs'
    # Only the neural model is substituted: no TensorFlow/GPU/training claim.
    monkeypatch.setattr(ensemble,'fit_predict_lstm',lambda **kwargs:np.zeros(kwargs['predict_end']-kwargs['predict_start']))
    monkeypatch.setattr(sys,'argv',['run','--data-dir',str(data),'--output-dir',str(output),
                                  '--tickers','TEST','--train-fraction','.75','--meta-fraction-of-train','0.3333333333333333'])
    run.main()
    splits=pd.read_csv(output/'splits_TEST.csv')
    assert list(splits['Rows'])==[60,30,30]
    assert list(pd.read_csv(output/'predictions_TEST.csv')['Date'])==list(market().index[90:].strftime('%Y-%m-%d'))
    assert (output/'backtests.csv').is_file() and (output/'TEST_model_comparison.png').is_file()
    assert set(pd.read_csv(output/'metrics.csv')['stacking_mode'])=={'chronological_holdout'}
