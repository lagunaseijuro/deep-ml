import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
    n = labels.shape[0]
    mse_values = []
    for epoch in range(epochs):
        prediction = features @ initial_weights + initial_bias


        logits = 1 / (1 + np.exp(-prediction))
        logits_diff = (logits - labels) * logits * (1 - logits)

        mse = np.mean((logits - labels) ** 2)
        mse_values.append(round(mse, 4))

        w_grad = (2 / n) * features.T @ logits_diff
        bias_grad = (2 / n) * np.sum(logits_diff)

        initial_weights -= learning_rate * w_grad
        initial_bias -= learning_rate * bias_grad

    return np.round(initial_weights, 4).tolist(), np.round(initial_bias, 4).tolist(), mse_values