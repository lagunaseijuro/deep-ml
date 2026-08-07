import numpy as np


def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    TPR, FPR = [0], [0]
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    unique_scores = np.unique(y_scores)
    unique_scores = np.sort(unique_scores)[::-1]

    for threshold in unique_scores:
        TP = np.sum((y_true == 1) & (y_scores >= threshold)) / np.sum(y_true)
        FP = np.sum((y_true == 0) & (y_scores >= threshold)) / np.sum(y_true == 0)

        TPR.append(TP)
        FPR.append(FP)


    return (FPR, TPR)

