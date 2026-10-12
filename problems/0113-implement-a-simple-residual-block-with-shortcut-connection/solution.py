import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	# Your code here
	identity = x
	output = np.fmax(x @ w1, 0)
	output = output @ w2

	return np.fmax(0, output + identity)