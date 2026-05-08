# models/trainer.py

from models.linearregression import get_model as get_linear
from models.ridge_model import get_model as get_ridge
from models.randomforest import get_model as get_rf
from models.gradientboost import get_model as get_gbm

MODELS = {
    'LinearRegression': get_linear,
    'Ridge':            get_ridge,
    'RandomForest':     get_rf,
    'GradientBoosting': get_gbm,
}

def train_all_models(X_train, y_train):
    """Train all models. Returns dict of fitted models."""
    trained = {}
    for name, get_model in MODELS.items():
        model = get_model()
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"  Trained: {name}")
    return trained

def predict_all_models(trained_models, X):
    """Get predictions from all trained models."""
    predictions = {}
    for name, model in trained_models.items():
        predictions[name] = model.predict(X)
    return predictions