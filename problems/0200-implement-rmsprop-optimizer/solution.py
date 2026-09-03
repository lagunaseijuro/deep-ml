import numpy as np

def rmsprop_update(params: list[float], grads: list[float], cache: list[float], 
                   lr: float = 0.01, beta: float = 0.9, epsilon: float = 1e-8) -> tuple[list[float], list[float]]:
	"""
	Perform RMSProp optimization update.
	
	Args:
		params: List of parameter values
		grads: List of gradients for each parameter
		cache: List of cache values (moving average of squared gradients)
		lr: Learning rate
		beta: Decay rate for moving average
		epsilon: Small constant for numerical stability
	
	Returns:
		Tuple of (updated_params, updated_cache)
	"""
	# Your code here
	params = np.asarray(params)
	cache = np.asarray(cache)
	grads = np.asarray(grads)

	cache = beta * cache + (1 - beta) * ((grads) ** 2)

	params = params - (lr / (epsilon + cache ** 0.5)) * grads

	return params.tolist(), cache.tolist()

	
	pass