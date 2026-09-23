import torch

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    # TODO: implement inverted dropout
    if not training:
        return x

    keep_prob = 1 - p
    mask = torch.bernoulli(torch.full_like(x, keep_prob))
    
    return x * mask / keep_prob
