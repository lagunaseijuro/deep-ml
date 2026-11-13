import numpy as np

def permutation_importance(X, y, weights, bias, permutations):
    X = np.asarray(X)

    result = []
    y_pred_baseline = X @ weights + bias
    r_baseline = 1 - np.sum((y - y_pred_baseline) ** 2) / (np.sum((y - np.mean(y)) ** 2))

    for idx in range(len(permutations)):
        r_sq = 0
        for jdx in range(len(permutations[idx])):
            X_perm = X.copy()
            X_perm[:, idx] = X[:, idx][permutations[idx][jdx]]

            y_pred = X_perm @ weights + bias

            r_sq += 1 - np.sum((y - y_pred) ** 2) / (np.sum((y - np.mean(y)) ** 2))


        r_sq /= len(permutations[idx])
        result.append(r_baseline - r_sq)

    return result