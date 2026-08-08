import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    n_samples = X.shape[0]
    for i in range(n_iter):
        idx = i % n_samples
        grad = 2 * (X[idx, :] @ weights - y[idx]) * X[idx, :].T 
        weights -= learning_rate * grad

    return weights
