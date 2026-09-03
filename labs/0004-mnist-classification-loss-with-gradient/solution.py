import numpy as np

def loss_function(preds: np.ndarray, target: np.ndarray, reduction: str = "mean", **kwargs):
    """
    preds:     [N, C] softmax probabilities (rows sum to 1)
    target:    [N]    class indices (int64)
    reduction: how to aggregate per-sample losses:
               "mean" → average over batch (gradient scaled by 1/N)
               "sum"  → sum over batch (gradient unscaled)
               "none" → return per-sample loss vector (no aggregation)
    **kwargs:  absorbs any extra arguments from the training harness

    Returns: (loss, grad) where grad has the same shape as preds
    """
    # Your implementation here
    N, C = preds.shape

    eps = 1e-12
    correct_probs = preds[np.arange(N), target]
    per_sample_loss = -np.log(correct_probs + eps)

    grad = np.zeros_like(preds)
    grad[np.arange(N), target] = - 1.0 / (correct_probs + eps)

    if reduction == 'mean':
        return np.mean(per_sample_loss), grad / N
    if reduction == 'sum':
        return np.sum(per_sample_loss), grad

    return per_sample_loss, grad