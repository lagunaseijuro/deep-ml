import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class TinyCNN(nn.Module):
    """Small CNN: Conv2d -> ReLU -> pool -> (optional extras) -> Linear."""

    def __init__(self, img_size=8, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=2)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(8, 16, kernel_size=2)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 8, kernel_size=2)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Linear(8, n_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))

        x = self.pool(self.conv2(x))

        x = self.relu(self.conv3(x))

        x = self.gap(x)

        x = x.squeeze(-1).squeeze(-1)

        return self.head(x)


def build_model(img_size=8, n_classes=2):
    return TinyCNN(img_size, n_classes)


def train_model(model, train_x, train_y, epochs=15, lr=0.01, batch_size=32, seed=0):
    torch.manual_seed(seed)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model





