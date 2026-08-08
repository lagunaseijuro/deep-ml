import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    weights_grad = 2 / (len(batch_indices)) * X[batch_indices].T @ (X[batch_indices] @ weights + bias - y[batch_indices])
    bias_grad = (2 / (len(batch_indices)) * np.sum((X[batch_indices] @ weights + bias - y[batch_indices])))
    
    bias -= lr * bias_grad
    weights -= lr * weights_grad

    result = weights.tolist()
    result.append(bias)

    return result
