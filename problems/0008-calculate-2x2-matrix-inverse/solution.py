import numpy as np
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    if np.linalg.det(np.array(matrix)) == 0:
        return None
    return np.linalg.inv(np.array(matrix))