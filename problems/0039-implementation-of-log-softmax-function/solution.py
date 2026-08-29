import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	scores = np.asarray(scores)

	softmax = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)))

	return np.log(softmax)

