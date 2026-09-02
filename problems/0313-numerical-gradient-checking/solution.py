import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    num_grad = []
    for i in range(len(x)):
        unit = np.array([0] * len(x))
        unit[i] = 1
        
        x_plus = x + epsilon * unit
        x_minus = x - epsilon * unit

        n_grad = (f(x_plus) - f(x_minus)) / (2 * epsilon)

        num_grad.append(n_grad)
    
    num_grad = np.asarray(num_grad)

    if np.linalg.norm(num_grad) + np.linalg.norm(analytical_grad) == 0:
        return num_grad, 0

    return num_grad, np.linalg.norm(num_grad - analytical_grad) / (np.linalg.norm(num_grad) + np.linalg.norm(analytical_grad)) 
