import numpy as np

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	labels = [0] * len(logits)
	labels[target] = 1
	labels = np.asarray(labels)

	logits = np.asarray(logits)

	probabilities = np.exp(logits - np.max(logits)) / (np.sum(np.exp(logits - np.max(logits))))

	return (probabilities - labels).tolist()