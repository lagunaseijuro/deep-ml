import numpy as np


def batch_normalization(
        X: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        running_mean: np.ndarray = None,
        running_var: np.ndarray = None,
        momentum: float = 0.1,
        epsilon: float = 1e-5,
        training: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Batch Normalization on BCHW input.

    Args:
        X: Input array of shape (B, C, H, W)
        gamma: Scale parameter of shape (1, C, 1, 1)
        beta: Shift parameter of shape (1, C, 1, 1)
        running_mean: Running mean for inference, shape (1, C, 1, 1)
        running_var: Running variance for inference, shape (1, C, 1, 1)
        momentum: Momentum for updating running statistics (following PyTorch convention)
        epsilon: Small constant for numerical stability
        training: If True, use batch statistics; if False, use running statistics

    Returns:
        Tuple of (normalized_output, updated_running_mean, updated_running_var)
    """
    # Your code here
    if not training:
        X_scaled = (X - running_mean) / np.sqrt(running_var + epsilon)
        return (gamma * X_scaled + beta, running_mean, running_var)

    n_channels = X.shape[1]

    running_mean = np.zeros(shape=(1, n_channels, 1, 1)) if running_mean is None else running_mean
    running_var = np.ones(shape=(1, n_channels, 1, 1)) if running_var is None else running_var

    x_mu = np.mean(X, axis=(0, 2, 3), keepdims=True)
    x_var = np.var(X, axis=(0, 2, 3), keepdims=True)
    X_scaled = (X - x_mu) / np.sqrt(x_var + epsilon)
    normalized_output = gamma * X_scaled + beta

    running_mean = (1 - momentum) * running_mean + momentum * x_mu
    running_var = (1 - momentum) * running_var + momentum * x_var

    return (normalized_output, running_mean, running_var)









