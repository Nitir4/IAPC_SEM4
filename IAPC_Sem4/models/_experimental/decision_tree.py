from sklearn.tree import DecisionTreeRegressor

def get_model():
    return DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=20,
    min_samples_split=30,
    random_state=42,
)