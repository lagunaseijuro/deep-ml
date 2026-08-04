import numpy as np

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    means = np.mean(X_train, axis=0)
    stds = np.std(X_train, axis=0)
    stds = np.where(stds == 0, 1, stds)

    return (X_test - means) / stds
