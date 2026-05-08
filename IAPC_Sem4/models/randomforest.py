from sklearn.ensemble import RandomForestRegressor
from config import RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF, RF_N_ESTIMATORS

def get_model():
    return RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=42,
        n_jobs=-1
    )