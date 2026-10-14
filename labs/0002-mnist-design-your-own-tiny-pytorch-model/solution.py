import torch, torch.nn as nn

def build_model() -> nn.Module:
    """
    Return a tiny nn.Module for MNIST classification (10 classes).
    IMPORTANT: If total trainable params > 2048, final accuracy will be set to 0.
    Tip: Consider very small convs, global average pooling, and tiny linear head.
    """
    class TinyNet(nn.Module):
         def __init__(self):
             super().__init__()

             self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
             self.relu = nn.ReLU()

             self.conv2 = nn.Conv2d(8, 20, kernel_size=3)
             self.pool = nn.MaxPool2d(2, 2)

             self.gap = nn.AdaptiveAvgPool2d(1)
             
             self.head = nn.Linear(20, 10)

         def forward(self, x):
            x = self.relu(self.conv1(x))
            
            x = self.pool(self.conv2(x))
            
            x = self.gap(x)
            x = x.squeeze(-1).squeeze(-1)

            return self.head(x)
             
    return TinyNet()
