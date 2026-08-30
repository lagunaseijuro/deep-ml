import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    # Your code here
    features = np.asarray(features)
    labels = np.asarray(labels)
    weights = np.asarray(weights)

    prediction = features @ weights + bias

    probabilities = 1 / (1 + np.exp(-prediction))

    mse = np.mean((probabilities - labels) ** 2)

    return probabilities, mse