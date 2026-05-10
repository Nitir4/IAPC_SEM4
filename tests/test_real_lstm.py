"""Small real-TensorFlow acceptance check; optional in lightweight installs."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import numpy as np
import pandas as pd
import pytest
from paper_reproduction.data import add_technical_features,build_target
from paper_reproduction.lstm import fit_predict_lstm


def test_real_tensorflow_training_and_prediction_are_finite():
    pytest.importorskip('tensorflow')
    rng=np.random.default_rng(505)
    close=100*np.exp(np.cumsum(rng.normal(0,.01,100)))
    frame=pd.DataFrame({'Close':close,'Open':close*.998,'High':close*1.01,'Low':close*.99,'Volume':1000.},index=pd.date_range('2020-01-01',periods=100,freq='B'))
    frame=add_technical_features(frame)
    forecast=fit_predict_lstm(frame,build_target(frame),train_end=70,predict_start=70,predict_end=80,
                             look_back=10,units=4,epochs=1,batch_size=16,seed=505)
    assert forecast.shape==(10,) and np.isfinite(forecast).all()
