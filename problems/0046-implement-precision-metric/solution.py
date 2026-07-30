import numpy as np
def precision(y_true, y_pred):
	acc = 0
	for idx in range(len(y_true)):
		if y_pred[idx] == 1 and y_true[idx] == 1:
			acc += 1
	return acc / np.sum(y_pred)
