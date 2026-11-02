import numpy as np

def explained_variance_ratio(X):
    """
    Calculate the explained variance ratio for PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
    
    Returns:
        List of explained variance ratios sorted in descending order
    """
    # Your code here
    X = np.asarray(X)

    X_scaled = X - np.mean(X, axis=0)

    cov_matrix = np.cov(X_scaled, rowvar=False, bias=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    eigenvalues = eigenvalues / np.sum(eigenvalues)

    return np.sort(eigenvalues)[::-1]