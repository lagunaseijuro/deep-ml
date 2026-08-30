import torch

def bce_with_logits(logits, targets):
    return round(torch.mean(torch.max(logits, torch.zeros(logits.shape[0])) - logits * targets + torch.log(1 +
    torch.exp(-torch.abs(logits))
    )).item(), 4)