import numpy as np

def smote(X_minority: np.ndarray, n_synthetic: int, k: int = 5) -> np.ndarray:
    result = []
    n_samples, n_features = X_minority.shape
    k = min(k, n_samples - 1)
    if k == 0 or n_synthetic == 0:
        return np.empty((0, n_features))
    for _ in range(n_synthetic):
        i = np.random.randint(0, n_samples)
        x_i = X_minority[i]
        neighbours = []
        for j in range(n_samples):
            if j == i:
                continue
            neighbours.append((np.linalg.norm(x_i - X_minority[j]), j))
        neighbours.sort()
        samples = neighbours[:k]
        x_nn = X_minority[samples[np.random.randint(0, k)][1]]
        gap = np.random.random()
        x_synthetic = x_i + gap * (x_nn - x_i)
        result.append(x_synthetic)
    result = np.array(result)
    return result
