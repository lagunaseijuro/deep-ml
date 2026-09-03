import torch
from torch.utils.data import TensorDataset, DataLoader

def batch_stats(X, y):
    dataset = TensorDataset(X, y)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    first_batch_X, first_batch_y = next(iter(dataloader))

    num_batches = len(dataloader)

    first_batch_X_shape_tuple = first_batch_X.shape

    return num_batches, tuple(first_batch_X_shape_tuple)
