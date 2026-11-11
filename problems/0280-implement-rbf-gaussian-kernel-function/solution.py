import numpy as np

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
    X2_norm = np.sum(X2 ** 2, axis=1)

    dot_product = X1 @ X2.T

    return np.exp(-gamma * (X1_norm + X2_norm - 2 * dot_product))


    