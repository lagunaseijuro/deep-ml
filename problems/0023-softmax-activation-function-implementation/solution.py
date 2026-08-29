import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    scores = np.array(scores)

    result = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)))

    return result.tolist()