import torch

def linear_backward(grad_output, x, W):
    grad_output = torch.tensor(grad_output)
    x = torch.tensor(x)
    W = torch.tensor(W)


    grad_input = grad_output @ W 

    grad_W = grad_output.T @ x

    grad_b = torch.sum(grad_output, dim=0)

    return grad_input, grad_W, grad_b
