import numpy as np

def pca_reconstruction_error(X: np.ndarray, n_components: int) -> float:
    """
    Compute the mean squared reconstruction error from PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Number of principal components to keep
        
    Returns:
        The mean squared reconstruction error (float)
    """
    n_samples = X.shape[0]
    X_centered = np.asarray(X - np.mean(X, axis=0))

    cov_matrix = (X_centered.T @ X_centered) / n_samples

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvectors = eigenvectors[:, :n_components]

    X_projected = X_centered @ eigenvectors

    X_reconstructed = X_projected @ eigenvectors.T

    return np.mean((X_centered - X_reconstructed) ** 2)
    