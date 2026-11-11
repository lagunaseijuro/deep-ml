import numpy as np

def svm_margin_width(w: np.ndarray) -> float:
    return 2 / np.linalg.norm(w)