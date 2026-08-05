import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)

	mse_error = 0.5 * (y_true - y_pred) ** 2
	mae_error = delta * (np.abs(y_true - y_pred) - 0.5 * delta)

	error = np.where(np.abs(y_true - y_pred) <= delta, mse_error, mae_error)

	return np.mean(error)