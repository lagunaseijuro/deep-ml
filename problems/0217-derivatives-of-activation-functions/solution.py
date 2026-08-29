import numpy as np

def sigmoid_func(x):
	return 1 / (1 + np.exp(-x))

def tanh_func(x):
	return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def activation_derivatives(x: float) -> dict[str, float]:

	result = {'sigmoid' : (sigmoid_func(x) * (1 - sigmoid_func(x))),
	'tanh' : 1 - (tanh_func(x)) ** 2, 
	'relu' : 1 if x > 0 else 0
	}

	return result