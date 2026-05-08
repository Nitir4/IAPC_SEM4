from sklearn.svm import SVR

def get_model():
    return SVR(
    kernel="rbf",
    C=0.1,
    epsilon=0.1,
)