import numpy as np

def to_categorical(x, n_col=None):
	if n_col is None:
		n_col = np.unique(x).size
	n = x.shape[0]
	result = np.zeros((n, n_col))
	for idx in range(n):
		result[idx][x[idx]] = 1
	return result