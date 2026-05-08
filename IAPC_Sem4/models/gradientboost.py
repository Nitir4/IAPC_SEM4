from sklearn.ensemble import GradientBoostingRegressor
from config import GBM_MAX_DEPTH, GBM_N_ESTIMATORS, GBM_LEARNING_RATE

def get_model():
    return GradientBoostingRegressor(
    n_estimators=50,      # down from whatever you have
    max_depth=2,          # strict
    learning_rate=0.03,   # slow
    subsample=0.7,
    min_samples_leaf=15
)