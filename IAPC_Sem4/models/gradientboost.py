from sklearn.ensemble import GradientBoostingRegressor

def get_model():
    return GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1
    )