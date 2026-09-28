import numpy as np

def bias_variance_decomp(predictions, y_true):
    """
    Compute the empirical bias-variance decomposition from bootstrap predictions.

    Args:
        predictions: array-like of shape (B, M) - predictions from B models at M test points
        y_true: array-like of shape (M,) - true target values

    Returns:
        dict with keys 'bias_squared', 'variance', 'mse'
    """
    predictions = np.asarray(predictions)
    y_true = np.asarray(y_true)

    MSE_ = float(np.mean(np.mean((predictions - y_true) ** 2, axis=0)))
    BIAS_SQ = float(np.mean((np.mean(predictions, axis=0) - y_true) ** 2))
    VAR = np.mean(np.mean((predictions - np.mean(predictions, axis=0)) ** 2, axis=0))

    return {
        'bias_squared' : BIAS_SQ,
        'variance' : VAR,
        'mse' : MSE_
    }

