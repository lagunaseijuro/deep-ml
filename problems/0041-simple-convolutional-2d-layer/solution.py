import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape

	# Your code here
	if padding > 0:
		input_pad = np.pad(input_matrix, ((padding, padding), (padding, padding)), 'constant', constant_values=0)
	else:
		input_pad = input_matrix
	
	output_height = (input_height + 2 * padding - kernel_height) // stride + 1
	output_width = (input_width + 2 * padding - kernel_width) // stride + 1
    
	output_matrix = np.zeros((output_height, output_width))

	for idx in range(output_height):
		for jdx in range(output_width):
			row_start = idx * stride
			row_end = row_start + kernel_height
			col_start = jdx * stride
			col_end = col_start + kernel_width

			window = input_pad[row_start:row_end, col_start:col_end]

			output_matrix[idx, jdx] = np.sum(window * kernel)


	return output_matrix
