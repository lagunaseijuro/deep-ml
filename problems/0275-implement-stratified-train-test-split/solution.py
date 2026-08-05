import numpy as np

def stratified_train_test_split(X, y, test_size, random_seed=None):
    np.random.seed(random_seed)

    train_indices = []
    test_indices = []

    classes = np.unique(y)

    for cls in classes:
        cls_indices = np.where(y == cls)[0]

        np.random.shuffle(cls_indices)

        n_samples = int(len(cls_indices) * test_size)

        test_indices.extend(cls_indices[:n_samples])
        train_indices.extend(cls_indices[n_samples:])
    
    X_train, X_test, y_train, y_test = \
    X[train_indices], X[test_indices], y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test



        

