import torch

def grad_of_quadratic(x_value: float) -> float:
    x_value = torch.tensor(x_value, dtype=torch.float32, requires_grad=True)

    f = x_value * x_value + 3 * x_value + 2

    f.backward()

    return x_value.grad.item()