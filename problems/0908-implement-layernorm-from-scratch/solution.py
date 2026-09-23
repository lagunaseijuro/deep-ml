import torch

def layer_norm(x, gamma, beta, eps=1e-5):
    # TODO: normalize over the last dim, then affine-transform with gamma and beta
    D = x.shape[-1]

    x_mu = torch.mean(x, dim=-1, keepdim=True)
    x_var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    
    X_scaled = (x - x_mu) / torch.sqrt(x_var + eps)
    
    return gamma * X_scaled + beta
