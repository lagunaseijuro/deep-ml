import numpy as np

import numpy as np
import copy


def elastic_net_gradient_descent(
        X: np.ndarray,
        y: np.ndarray,
        alpha1: float = 0.1,
        alpha2: float = 0.1,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-4,
) -> tuple:
    n_samples, n_features = X.shape
    y = y.reshape(-1, 1)

    weights = np.zeros(shape=(n_features, 1))
    bias = 0

    for _ in range(max_iter):
        w_grad = (1 / n_samples) * (X.T @ (X @ weights + bias - y)) + alpha1 * np.sign(weights) + 2 * \
        alpha2 * weights
        b_grad = np.mean(X @ weights + bias - y)

        if np.sum(np.abs(w_grad)) < tol:
            break

        weights -= learning_rate * w_grad
        bias -= learning_rate * b_grad

    return (weights.flatten().tolist(), bias)





