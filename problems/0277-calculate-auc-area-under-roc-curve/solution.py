import numpy as np

def calculate_auc(y_true, y_scores):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    auc_points = [(0, 0)]

    order = np.argsort(y_scores)[::-1]
    y_true = y_true[order]
    y_scores = y_scores[order]

    total_area = 0

    for threshold in y_scores:
        y_pred = (y_scores >= threshold)

        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))

        TPR = TP / (TP + FN) if TP + FP != 0 else 0
        FPR = FP / (FP + TN) if TP + FN != 0 else 0

        if (TPR, FPR) == auc_points[-1]:
            continue
        
        last_TPR, last_FPR = auc_points[-1]
        total_area += ((TPR + last_TPR) / 2) * (FPR - last_FPR)

        auc_points.append((TPR, FPR))

    last_FPR, last_TPR = auc_points[-1]
    total_area += ((1 + last_FPR) / 2) * (1 - last_TPR)

    return total_area
