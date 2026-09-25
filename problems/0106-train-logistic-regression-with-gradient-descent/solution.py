import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    n_samples, n_features = X.shape

    weights = np.zeros(shape=(n_features, 1))
    bias = 0
    losses = []

    for _ in range(iterations):
        logits = X @ weights + bias
        probs = 1 / (1 + np.exp(-logits))

        loss = -(y @ np.log(probs) + (1 - y) @ np.log(1 - probs))
        losses.append(loss.item())

        weights += learning_rate * (X.T @ (y.reshape(n_samples, 1) - probs))
        bias += learning_rate * (y.reshape(n_samples, 1) - probs).sum()

    params = np.concatenate([[bias], weights.ravel()])
    return np.round(params, 4).tolist(), losses