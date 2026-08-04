import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    n_samples, n_features = X.shape
    result = np.ones((n_samples, 1))
    for d in range(1, degree + 1):
        poly_indices = list(combinations_with_replacement(np.arange(n_features), d))
        for idx in poly_indices:
            result = np.hstack((result, np.prod(X[:, idx], axis=1, keepdims=True)))
    return np.sort(result, axis=1)