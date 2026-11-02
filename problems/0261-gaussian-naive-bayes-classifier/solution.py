import numpy as np

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
	n_samples, n_features = X_train.shape
	aprior_probs = []
	means, vrs = [], []

	for cls in np.unique(y_train):
		aprior_probs.append(np.log(np.sum(y_train == cls) / (n_samples + 1e-9)))
		means.append(np.mean(X_train[y_train == cls], axis=0))
		vrs.append(np.var(X_train[y_train == cls], axis=0))

	aprior_probs = np.asarray(aprior_probs)
	means = np.asarray(means)
	vrs = np.asarray(vrs)

	preds = []
	for cls in range(len(np.unique(y_train))):
		preds.append(
			np.sum(np.log(np.exp(-((X_test - means[cls]) ** 2) / (2 * vrs[cls])) / np.sqrt(2 * np.pi * vrs[cls])), axis=1) + aprior_probs[cls]
		)

	preds = np.asarray(preds)

	return np.argmax(preds, axis=0)


	



	