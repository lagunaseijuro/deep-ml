
import numpy as np

def r_squared(y_true, y_pred):
	nom = np.sum((y_true - y_pred) ** 2)
	denom = np.sum((y_true - np.mean(y_true)) ** 2)
	return 1 - (nom / denom)
