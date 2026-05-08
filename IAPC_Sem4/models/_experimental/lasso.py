from sklearn.linear_model import Lasso

def get_model():
    return Lasso(alpha=0.01)