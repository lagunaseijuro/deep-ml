
import numpy as np

def jaccard_index(y_true, y_pred):
	return round(np.sum((y_true == 1) & (y_pred == 1)) / np.sum((y_true == 1) | (y_pred == 1)), 3)

