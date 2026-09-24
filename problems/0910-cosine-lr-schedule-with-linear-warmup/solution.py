import math


def cosine_with_warmup(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    # TODO: piecewise linear warmup, then half-cosine decay
    if step > total_steps or step < 0:
        return 0.0

    if step <= warmup_steps:
        if warmup_steps == 0:
            return base_lr

        return base_lr * (step / warmup_steps)

    if total_steps == warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * (1 + math.cos(math.pi * progress)) / 2