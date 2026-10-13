import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # TODO: save anything backward will need, then return the ReLU output
        result = torch.relu(x)
        ctx.save_for_backward(x)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: return dL/dx using the saved tensors and grad_output
        x = ctx.saved_tensors[0]
        return grad_output * (x > 0)
