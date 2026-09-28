import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    def gini(y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)

        return 1 - np.sum(probs ** 2)

    n_samples, n_features = X.shape
    best_feature_idx = -1
    best_gini_split = np.inf
    best_threshold = None

    for feature in range(n_features):
        for threshold in np.unique(X[:, feature]):
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask
            y_left = y[left_mask]
            y_right = y[right_mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            left_gini = gini(y_left)
            right_gini = gini(y_right)

            G_split = len(y_left) / n_samples * left_gini + \
            len(y_right) / n_samples * right_gini

            if G_split < best_gini_split:
                best_feature_idx = feature
                best_gini_split = G_split
                best_threshold = threshold

    return (best_feature_idx, best_threshold)




    