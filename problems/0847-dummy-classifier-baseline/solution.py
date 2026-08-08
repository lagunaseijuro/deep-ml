import numpy as np
from collections import Counter

def dummy_classifier(y_train, n_test, strategy, constant=None):
    y = np.asarray(y_train)

    if strategy == 'constant':
        return [constant] * n_test

    unique_classes, counts = np.unique(y, return_counts=True)

    if strategy == 'most_frequent':
        best_class = unique_classes[np.argmax(counts)]
        return [best_class] * n_test

    if strategy == 'uniform':
        indices = np.arange(n_test) % len(unique_classes)
        return unique_classes[indices].tolist()

    native = n_test * (counts / (len(y)))
    main = np.floor(native).astype(int)
    fractional = native - main

    remaining = n_test - main.sum()
    if remaining > 0:

        order = np.argsort(-fractional, kind='mergesort')
        for i in range(remaining):
            main[order[i]] += 1

    result = np.repeat(unique_classes, main)
    return result.tolist()

