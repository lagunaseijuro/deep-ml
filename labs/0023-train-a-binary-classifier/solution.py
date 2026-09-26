import numpy as np

def train(X_train, y_train, X_val, y_val):
    c0 = X_train[y_train == 0].mean(axis=0)
    c1 = X_train[y_train == 1].mean(axis=0)

    def predict(X):
        d0 = np.linalg.norm(X - c0, axis=1)
        d1 = np.linalg.norm(X - c1, axis=1)
        return (d1 < d0).astype(int)

    return predict