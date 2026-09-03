import torch
import torch.nn as nn

def count_params():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )

    total = 0
    for param in model.parameters():
        if param.requires_grad:
            total += param.numel()

    return total
