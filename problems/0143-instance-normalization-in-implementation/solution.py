import numpy as np

def instance_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    x_mu = np.mean(X, axis=(2, 3), keepdims=True)
    x_var = np.var(X, axis=(2, 3), keepdims=True)
    
    X_scaled = (X - x_mu) / np.sqrt(x_var + epsilon)
    
    return gamma * X_scaled + beta


