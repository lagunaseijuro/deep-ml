import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    cols = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), cols, X)
    X_clean = X.copy()
    
    # TODO: Fill in NaN values
    
    return X_clean
