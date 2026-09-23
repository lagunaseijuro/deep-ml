import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    # Your code here
    B, n_channels, H, W = X.shape
    axis = (2, 3, 4)

    x_shaped = X.reshape(B, num_groups, n_channels // num_groups, H, W)

    x_mu = np.mean(x_shaped, axis=axis, keepdims=True)
    x_var = np.var(x_shaped, axis=axis, keepdims=True)

    x_scaled = (x_shaped - x_mu) / np.sqrt(x_var + epsilon)
    x_scaled = x_scaled.reshape(B, n_channels, H, W)
    
    return gamma * x_scaled + beta


