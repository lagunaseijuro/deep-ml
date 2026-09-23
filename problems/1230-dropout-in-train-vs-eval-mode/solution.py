import torch
import torch.nn as nn


def dropout_demo():
    torch.manual_seed(0)

    x = torch.ones(10)
    drop = nn.Dropout(p=0.5)

    drop.eval()
    eval_output = drop(x)

    drop.train()
    train_nonzero_count = drop(x).sum().item() / 2

    return (eval_output, int(train_nonzero_count))