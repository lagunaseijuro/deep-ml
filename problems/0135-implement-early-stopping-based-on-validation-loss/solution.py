from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    best_epoch = cnt = 0
    epoch_to_end = len(val_losses) - 1
    for idx in range(1, len(val_losses)):
        if val_losses[best_epoch] - val_losses[idx] > min_delta:
            best_epoch = idx
            cnt = 0
            continue
        cnt += 1
        if cnt == patience:
            epoch_to_end = idx 
            break 

    return epoch_to_end, best_epoch
        
        