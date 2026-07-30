
from collections import Counter
from sklearn.metrics import confusion_matrix as cm

def confusion_matrix(data):
	res = [[0, 0], [0, 0]]
	y_true, y_pred = zip(*data)
	for i in range(len(data)):
		tr, pr = y_true[i], y_pred[i]
		if tr == pr == 1:
			res[0][0] += 1
		elif tr == 1 and pr == 0:
			res[0][1] += 1
		elif tr != 1 and pr == 1:
			res[1][0] += 1
		else:
			res[1][1] += 1
	return res
