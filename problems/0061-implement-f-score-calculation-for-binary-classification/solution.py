import numpy as np

def f_score(y_true, y_pred, beta):
	TP = np.sum((y_true == 1) & (y_pred == 1))
	FN = np.sum((y_true == 1) & (y_pred == 0))
	FP = np.sum((y_true == 0) & (y_pred == 1))
	precision = TP / (TP + FP)
	recall = TP / (TP + FN)
	return round((1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall), 3)
	
