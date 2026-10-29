import numpy as np

def k_nearest_neighbors(points, query_point, k):
    return sorted(points, key=lambda x: np.linalg.norm(np.asarray(x) - np.asarray(query_point)))[:k]