from sklearn.tree import DecisionTreeRegressor

def get_model():
    return DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=10
    )