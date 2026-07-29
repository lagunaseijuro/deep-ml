import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Note: `f` is accepted only for interface parity with the PyTorch/Tinygrad
    # variants, which derive gradients from it via autograd. This version uses
    # `grad` only — the objective value `f` is never evaluated.
    # Your code here
    x0 = x0.reshape(-1, 1)
    n = x0.shape[0]
    m, v = np.zeros((n, 1)), np.zeros((n, 1))
    for t in range(1, num_iterations + 1):
        g = grad(x0)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_ = m / (1 - beta1 ** t)
        v_ = v / (1 - beta2 ** t)
        x0 = x0 - (learning_rate / (epsilon + (v_ ** 0.5))) * m_
    return x0[:, 0]
