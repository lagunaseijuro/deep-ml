import numpy as np
import math

def overlapping_max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """
    Applies overlapping max pooling to a 4D tensor (N, C, H, W).
    Uses ceil mode for output dimensions (allows partial windows at boundaries).

    Args:
        x: Input array of shape (N, C, H, W)
        kernel_size: Size of pooling window (int)
        stride: Stride between pooling windows (int), must be < kernel_size

    Returns:
        A 4D tensor after overlapping pooling with ceil mode.
    """
    # Your code here
    N, C, H, W = x.shape
    
    H_out = math.ceil((H - kernel_size) / stride) + 1
    W_out = math.ceil((W - kernel_size) / stride) + 1
    output_tensor = np.zeros(shape=(N, C, H_out, W_out), dtype=x.dtype)

    for sample in range(N):
        for channel in range(C):
            for idx in range(H_out):
                for jdx in range(W_out):
                    row_start = idx * stride 
                    row_end = min(row_start + kernel_size, H)
                    col_start = jdx * stride 
                    col_end = min(col_start + kernel_size, W)

                    window = x[sample, channel, row_start:row_end, col_start:col_end]

                    output_tensor[sample, channel, idx, jdx] = np.max(window)

    return output_tensor










