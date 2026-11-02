import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    data_scaled = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    cov_matrix = np.cov(data_scaled, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    signs = np.sign(eigenvectors[0, :])
    signs[signs == 0] = 1

    eigenvectors = signs * eigenvectors

    return np.round(eigenvectors[:, :k], 4)
    

    
