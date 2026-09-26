import numpy as np

def train(X, y, W, b):
    n = X.shape[0]
    X_aug = np.hstack([np.ones(shape=(n, 1)), X])
    solution = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y 

    b = solution[0]
    W = solution[1:]

    return W, b
