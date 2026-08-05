import numpy as np

def stratified_kfold_indices(y, n_splits):
	test_folds = [[] for _ in range(n_splits)]

	unique_classes = np.unique(y)

	for cls in unique_classes:
		cls_indices = np.where(y == cls)[0]

		base_size = len(cls_indices) // n_splits
		rem = len(cls_indices) % n_splits

		start_idx = 0
		for idx in range(n_splits):
			fold_size = base_size + (1 if idx < rem else 0)

			test_folds[idx].extend(cls_indices[start_idx:start_idx + fold_size].tolist())

			start_idx += fold_size
	
	result = []
	
	for idx in range(len(test_folds)):
		train_indices = np.array(list(set(np.arange(len(y))) - set(test_folds[idx])))
		test_indices = np.array(test_folds[idx])
		
		result.append([np.sort(train_indices), np.sort(test_indices)])
	
	return result
	