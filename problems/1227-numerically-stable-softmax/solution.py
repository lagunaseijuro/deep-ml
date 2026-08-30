import torch

def softmax(t, dim):
    exponents = torch.exp(t - torch.max(t, dim=dim, keepdim=True)[0])

    scores = exponents / torch.sum(exponents, dim=dim, keepdim=True)

    return scores