import numpy as np

def softmax_derivative(x: list[float]) -> list[list[float]]:
	scores = np.asarray(x)

	softmax = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)))
	softmax = softmax.flatten()

	return np.diag(softmax) - np.outer(softmax, softmax)