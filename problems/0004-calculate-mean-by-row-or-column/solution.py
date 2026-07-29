import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	m = 0 if mode == 'column' else 1
	return np.mean(np.array(matrix), m)