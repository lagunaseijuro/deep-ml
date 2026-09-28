import numpy as np

def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
	"""
	Apply weight decay (L2 regularization) to parameters.
	
	Args:
		parameters: List of parameter arrays
		gradients: List of gradient arrays
		lr: Learning rate
		weight_decay: Weight decay factor
		apply_to_all: Boolean list indicating which parameter groups get weight decay
	
	Returns:
		Updated parameters
	"""
	# Your code here
	parameters = np.asarray(parameters)
	gradients = np.asarray(gradients)
	apply_to_all = np.asarray(apply_to_all)


	w_mask = np.where(apply_to_all, weight_decay, 0.0)

	return (parameters - lr * (gradients +  w_mask * parameters)).tolist()