import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    diag = np.diag(predicted_probs @ true_labels.T)

    return -np.mean(np.log(diag + epsilon))