import torch

def cross_entropy(logits, targets):
    # TODO: numerically stable mean cross-entropy
    softmax = torch.exp(logits - torch.amax(logits, dim=-1, keepdim=True))
    all_preds = softmax / torch.sum(softmax, dim=-1, keepdim=True)

    preds = all_preds[torch.arange(logits.shape[0]), targets]
    return -torch.mean(torch.log(preds))