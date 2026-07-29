def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	if len(a[0]) != len(b):
		return -1
	res = [0] * len(a)
	for row in range(len(a)):
		for col in range(len(a[0])):
			res[row] += a[row][col] * b[col]
	return res
