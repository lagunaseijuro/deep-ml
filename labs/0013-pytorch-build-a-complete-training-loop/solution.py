import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a PyTorch model and return training history.
    
    This is the standard PyTorch training pattern you'll use everywhere.
    Now you can use torch.optim to handle the gradient updates!
    
    Args:
        model: nn.Module to train
        X_train: training features, shape (N, ...)
        y_train: training labels, shape (N,)
        X_val: validation features, shape (M, ...)
        y_val: validation labels, shape (M,)
        epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate
    
    Returns:
        history: List of dicts, one per epoch, with keys:
            - 'epoch': epoch number (starting from 1)
            - 'train_loss': average training loss for the epoch
            - 'val_loss': validation loss after the epoch
            - 'val_accuracy': validation accuracy after the epoch
    
    Steps:
        1. Create optimizer: optim.Adam(model.parameters(), lr=lr)
        2. Create loss function: nn.CrossEntropyLoss()
        3. For each epoch:
            a. Shuffle training data
            b. Loop over mini-batches:
                - optimizer.zero_grad()
                - Forward pass
                - Compute loss
                - loss.backward()
                - optimizer.step()
            c. Compute validation accuracy
            d. Append metrics to history
        4. Return history
    
    Hints:
        - torch.randperm(n) gives a random permutation for shuffling
        - Use model.train() before training, model.eval() before validation
        - Use torch.no_grad() during validation
        - logits.argmax(dim=1) gives predicted classes
    """
    # TODO: Implement the training loop
    
    history = []
    
    # Your code here
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        num_batches = 0

        perm = torch.randperm(X_train.size(0))
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        for idx in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[idx: idx + batch_size]
            y_batch = y_shuffled[idx: idx + batch_size]

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            num_batches += 1
            total_loss += loss.item()

        avg_total_loss = total_loss / num_batches

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)

            predictions = val_output.argmax(dim=1)
            correct = (predictions == y_val).sum().item()
            val_accuracy = correct / len(y_val)

        history.append({
            'epoch': epoch,
            'train_loss': avg_total_loss,
            'val_loss': val_loss.item(),
            'val_accuracy': val_accuracy
        })

    return history
