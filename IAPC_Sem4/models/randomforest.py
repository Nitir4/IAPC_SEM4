from sklearn.ensemble import RandomForestRegressor

def get_model():
    return RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=10,
    min_samples_split=20,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)