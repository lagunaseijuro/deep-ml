import torch
import torch.nn as nn


def two_layer_mlp_forward(x, w1, b1, w2, b2):
    net = nn.Sequential(
        nn.Linear(2, 2),
        nn.ReLU(),
        nn.Linear(2, 1)
    )

    with torch.no_grad():
        net[0].weight.copy_(w1)
        net[0].bias.copy_(b1)

        net[2].weight.copy_(w2)
        net[2].bias.copy_(b2)

    output = net(x)

    return output.item()