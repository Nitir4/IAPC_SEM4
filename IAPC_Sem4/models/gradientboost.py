from sklearn.ensemble import GradientBoostingRegressor
from config import GBM_MAX_DEPTH, GBM_N_ESTIMATORS, GBM_LEARNING_RATE

def get_model():
    return GradientBoostingRegressor(
        n_estimators=GBM_N_ESTIMATORS,
        max_depth=GBM_MAX_DEPTH,
        learning_rate=GBM_LEARNING_RATE,
        subsample=0.7,
        min_samples_leaf=15,
        random_state=42
    )