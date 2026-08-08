import numpy as np

def dummy_regressor(y_train, n_test, strategy='mean', constant=None, quantile=None):
    y = np.asarray(y_train)

    if strategy == 'mean':
        return [np.mean(y)] * n_test

    if strategy == 'median':
        return [np.median(y)] * n_test

    if strategy == 'constant' and constant is not None:
        return [constant] * n_test
    elif strategy == 'constant' and constant is None:
        raise ValueError

    if quantile is None:
        raise ValueError
    q = np.quantile(y, quantile)

    return [q] * n_test