import numpy as np

def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    """
    Вычисляет метрики для бинарной классификации.
    """
    # Превращаем в массивы для удобства
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # 1. Считаем TP, TN, FP, FN
    TP = np.sum((actual == 1) & (predicted == 1))
    TN = np.sum((actual == 0) & (predicted == 0))
    FP = np.sum((actual == 0) & (predicted == 1))
    FN = np.sum((actual == 1) & (predicted == 0))
    
    # 2. Матрица ошибок (2x2)
    # FIXED: Standard sklearn layout [[TP, FN], [FP, TN]]
    confusion_matrix = [
        [int(TP), int(FN)], 
        [int(FP), int(TN)]
    ] 
    
    # 3. Accuracy (точность)
    total = TP + TN + FP + FN
    accuracy = float((TP + TN) / total) if total > 0 else 0.0
    
    # 4. F1 Score 
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = float(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0)
    
    # 5. Specificity (True Negative Rate)
    specificity = float(TN / (TN + FP) if (TN + FP) > 0 else 0)
    
    # 6. Negative Predictive Value (NPV)
    negative_predictive_value = float(TN / (TN + FN) if (TN + FN) > 0 else 0)
    
    return (confusion_matrix, accuracy, f1_score, specificity, negative_predictive_value)