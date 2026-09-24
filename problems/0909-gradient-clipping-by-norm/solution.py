import torch

def clip_grad_norm(parameters, max_norm: float) -> float:
    # TODO: compute total grad norm, scale in-place if it exceeds max_norm, return original norm
    parameters = list(parameters)

    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            total_norm += param.grad.detach().norm().item() ** 2

    total_norm = total_norm ** 0.5

    if total_norm > max_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad.mul_(max_norm / total_norm)

    return float(total_norm)
