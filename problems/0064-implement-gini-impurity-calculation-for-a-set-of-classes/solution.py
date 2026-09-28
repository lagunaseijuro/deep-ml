
import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	_, counts = np.unique(y, return_counts=True)
	counts = (counts / len(y)) ** 2
	val = 1 - np.sum(counts)

	return round(val,3)