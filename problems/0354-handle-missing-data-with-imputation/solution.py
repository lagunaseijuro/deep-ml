import numpy as np
from scipy.stats import mode

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    cols = None
    if strategy == 'mean':
        cols = np.nanmean(data, axis=0)
    elif strategy == 'median':
        cols = np.nanmedian(data, axis=0)
    else:
        cols = mode(data, nan_policy='omit', axis=0).mode
    nan_mask = np.isnan(data)
    data = np.where(nan_mask, cols, data)
    return data
