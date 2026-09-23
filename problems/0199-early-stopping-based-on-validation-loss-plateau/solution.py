def early_stopping(val_losses: list[float], patience: int = 5, min_delta: float = 0.0) -> list[bool]:
    """
    Determine at each epoch whether training should stop based on validation loss.

    Args:
        val_losses: List of validation losses at each epoch
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in validation loss to qualify as improvement

    Returns:
        List of booleans indicating whether to stop at each epoch
    """
    # Your code here
    best_epoch = cnt = 0
    epoch_to_end = None
    result = [False] * len(val_losses)

    for idx in range(1, len(val_losses)):
        if val_losses[best_epoch] - val_losses[idx] > min_delta + 1e-9:
            best_epoch = idx
            cnt = 0
            continue
        cnt += 1
        if cnt >= patience:
            epoch_to_end = idx
            result[epoch_to_end] = True

    return result