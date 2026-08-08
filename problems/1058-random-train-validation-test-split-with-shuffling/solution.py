import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    n_samples = data.shape[0]

    indices = np.random.default_rng(seed).permutation(n_samples)

    train_end = int(n_samples * train_frac)
    validation_end = train_end + int(n_samples * validation_frac)

    result = []

    train_indices = indices[:train_end]
    val_indices = indices[train_end:validation_end]
    test_indices = indices[validation_end:]

    result.append(data[train_indices])
    result.append(data[val_indices])
    result.append(data[test_indices])

    return result