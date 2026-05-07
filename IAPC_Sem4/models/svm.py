from sklearn.svm import SVR

def get_model():
    return SVR(
        kernel="rbf",
        C=100,
        epsilon=0.1
    )