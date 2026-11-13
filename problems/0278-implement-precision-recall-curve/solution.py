import numpy as np

def precision_recall_curve(y_true: list, y_scores: list) -> tuple:
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    precisions = []
    recalls = []

    order = np.argsort(y_scores)[::-1]
    y_scores = y_scores[order]
    y_true = y_true[order]

    thresholds = []

    for threshold in y_scores:
        if threshold in thresholds:
            continue
        
        thresholds.append(threshold)

        y_pred = (y_scores >= threshold)

        TP = np.sum(((y_true == 1) & (y_pred == 1)))
        FP = np.sum(((y_true == 0) & (y_pred == 1)))
        FN = np.sum(((y_true == 1) & (y_pred == 0)))

        precision = TP / (TP + FP) if TP + FP != 0 else 0.0
        recall = TP / (TP + FN) if TP + FN != 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    
    return precisions, recalls, thresholds











    
    