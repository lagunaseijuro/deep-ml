import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    m, n = X.shape
    weights = weights.reshape(-1, 1)
    y = y.reshape(-1, 1)
    if method == 'stochastic':
        for epoch in range(n_epochs):
            for sample in range(m):
                weights -= learning_rate * 2 * X.T[:, sample:sample + 1] * (X[sample:sample + 1, :] @ weights - y[sample:sample + 1])
    elif method == 'mini_batch':
        for epoch in range(n_epochs):
            for sample in range(0, m, batch_size):
                weights -= learning_rate * (2 / batch_size) * X.T[:, sample:sample + batch_size] @ (X[sample:sample + batch_size, :] @ weights - y[sample:sample + batch_size])
    else:
        for epoch in range(n_epochs):
            weights -= learning_rate * (2 / m) * X.T @ (X @ weights - y)
    return weights[:, 0]
