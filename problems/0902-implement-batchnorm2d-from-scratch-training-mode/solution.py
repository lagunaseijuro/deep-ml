import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    # TODO: training-mode batchnorm2d
    B, n_channels, H, W = x.shape

    x_mu = torch.mean(x, dim=(0, 2, 3), keepdim=True)
    x_var = torch.var(x, dim=(0, 2, 3), keepdim=True, unbiased=False)

    x_scaled = (x - x_mu) / torch.sqrt(x_var + eps)

    gamma = torch.reshape(gamma, shape=(1, n_channels, 1, 1))
    beta = torch.reshape(beta, shape=(1, n_channels, 1, 1))

    return gamma * x_scaled + beta

