import numpy as np


def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	z = X @ weights + bias
	res = []
	for i in range(len(z)):
		if 1 / (1 + np.exp(-z[i])) >= 0.5:
			res.append(1)
		else:
			res.append(0)
	return res