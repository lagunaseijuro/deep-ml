import numpy as np

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	return np.sort(np.linalg.eigvals(np.array(matrix)))[::-1].tolist()