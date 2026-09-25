import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """Split rows into train/val/test folds, fit something on the training fold, predict the test fold.

    Returns
    -------
    test_predictions : np.ndarray, shape (len(test_idx),)
    train_idx, val_idx, test_idx : 1-D integer arrays forming a partition of range(len(y))
    """
    np.random.seed(seed)

    n_samples, n_features = X.shape

    indices = np.random.permutation(np.arange(n_samples))
    X_shuffled, y_shuffled = X[indices], y[indices]

    first_border = int(n_samples * train_frac)
    second_border = int(first_border + n_samples * val_frac)

    X_train, y_train = X_shuffled[:first_border], y_shuffled[:first_border]
    X_val, y_val = X_shuffled[first_border:second_border], y_shuffled[first_border:second_border]
    X_test, y_test = X_shuffled[second_border:], y_shuffled[second_border:]

    test_predictions = np.full_like(y_test, np.median(y_train))

    train_idx, val_idx, test_idx = indices[:first_border], indices[first_border:second_border], indices[second_border:]

    return test_predictions, train_idx, val_idx, test_idx


