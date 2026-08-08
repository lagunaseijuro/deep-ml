import numpy as np

def min_max(x: list[float]) -> list[float]:
    x = np.array(x)

    x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))

    return x.tolist()