import torch

def mse(pred, target):
    result = torch.mean((target - pred) ** 2)

    return result.item()