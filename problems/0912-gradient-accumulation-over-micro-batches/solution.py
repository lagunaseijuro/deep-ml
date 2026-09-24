import torch

def accumulated_step(model, micro_batches, optimizer, criterion):
    # TODO: zero grads, accumulate over micro-batches with proper scaling, step once, return mean loss
    optimizer.zero_grad()
    N = len(micro_batches)
    total_loss = 0

    for x, y in micro_batches:
        preds = model(x)
        loss = criterion(preds, y)
        (loss / N).backward()
        total_loss += loss.item()

    optimizer.step()
    total_loss /= N

    return float(total_loss)


