import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Генерирует индексы для K-Fold Cross-Validation.
    
    Параметры:
    n_samples (int): Общее количество объектов в датасете
    k (int): Количество фолдов (по умолчанию 5)
    shuffle (bool): Перемешивать ли индексы перед разбиением (по умолчанию True)
    
    Возвращает:
    list of tuples: Список из k кортежей (train_indices, test_indices)
    """
    # 1. Создаем список индексов [0, 1, 2, ..., n_samples-1]
    indices = np.arange(n_samples).tolist()
    
    # 2. Перемешиваем, если нужно
    if shuffle:
        np.random.shuffle(indices)
    
    # 3. Определяем размер каждого фолда
    fold_sizes = []
    base_size = n_samples // k          # Минимальный размер фолда
    remainder = n_samples % k           # Сколько фолдов получат +1 объект
    
    for i in range(k):
        if i < remainder:
            fold_sizes.append(base_size + 1)  # Первые фолды получают лишние объекты
        else:
            fold_sizes.append(base_size)       # Остальные - базовый размер
    
    # 4. Создаем фолды (списки индексов для каждого фолда)
    folds = []
    start_idx = 0
    for size in fold_sizes:
        folds.append(indices[start_idx:start_idx + size])
        start_idx += size
    
    # 5. Для каждого фолда формируем пары (train, test)
    result = []
    for test_idx in range(k):
        test_indices = folds[test_idx]
        
        # Собираем все остальные фолды в тренировочный набор
        train_indices = []
        for train_idx in range(k):
            if train_idx != test_idx:
                train_indices.extend(folds[train_idx])
        
        result.append((train_indices, test_indices))
    
    return result