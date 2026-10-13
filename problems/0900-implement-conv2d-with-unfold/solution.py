import torch
import torch.nn.functional as F
import math

def conv2d_via_unfold(x, W, b, stride=1, padding=0):
    # TODO: implement conv2d using F.unfold + matmul
    N, C_in, H_in, W_in = x.shape
    C_out, _, kH, kW = W.shape

    H_out = math.floor((H_in + 2 * padding - kH)/ stride) + 1
    W_out = math.floor((W_in + 2 * padding - kW) / stride) + 1

    windows_as_cols = F.unfold(x, kernel_size=(kH, kW), stride=stride, padding = padding) # (N, C_in * kH * kW, H_out * W_out)

    W = W.reshape(C_out, -1)

    output = torch.matmul(W, windows_as_cols)
    output = output.reshape(N, C_out, H_out, W_out)

    return output + b.view(1, C_out, 1, 1)


    
