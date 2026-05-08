from sklearn.ensemble import GradientBoostingRegressor

def get_model():
    return GradientBoostingRegressor(
    n_estimators=50,
    learning_rate=0.03,
    max_depth=2,
    subsample=0.7,
    random_state=42,
)