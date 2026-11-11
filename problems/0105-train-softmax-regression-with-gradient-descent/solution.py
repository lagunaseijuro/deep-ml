import numpy as np

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    n_samples, n_features = X.shape
    n_classes = np.max(y) + 1
    X = np.hstack([np.ones((n_samples, 1)), X])

    P_one_hot = np.zeros((n_samples, n_classes))
    P_one_hot[np.arange(n_samples), y] = 1

    W = np.zeros((n_classes, n_features + 1))

    losses = []
    for _ in range(iterations):
        raw_logits = X @ W.T
        raw_logits_stable = np.exp(raw_logits - np.max(raw_logits, axis=1, keepdims=True))
        probs = raw_logits_stable / np.sum(raw_logits_stable, axis=1, keepdims=True)

        loss = -np.sum(P_one_hot * np.log(probs + 1e-15))
        losses.append(loss)

        grad = (probs - P_one_hot).T @ X

        W -= learning_rate * grad

    return np.round(W, 4), np.round(losses, 4)


