from sklearn.ensemble import RandomForestRegressor

def get_model():
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        random_state=42
    )