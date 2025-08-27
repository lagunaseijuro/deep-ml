import math
import numpy as np

def sigmoid(z: float) -> float:
	#Your code here
	result = np.round(1 / (1 + np.exp(-z)), 4)
	return result