import numpy as np

def ordinal_encode(X, categories):
    n_samples, n_features = len(X), len(X[0])
    result = np.zeros((n_samples, n_features))

    for col in range(n_features):
        mappings = {val: idx for idx, val in enumerate(categories[col])}

        for row in range(n_samples):
            result[row, col] = mappings.get(X[row][col], -1)
    return result

