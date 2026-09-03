import torch

def train_step(model, x, y, optimizer, loss_fn):
    optimizer.zero_grad()

    pred = model(x)

    loss = loss_fn(pred, y)

    loss.backward()

    optimizer.step()

    return loss.item()
