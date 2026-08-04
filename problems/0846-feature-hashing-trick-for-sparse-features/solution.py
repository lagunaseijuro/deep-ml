import hashlib

def feature_hash(features: dict, n_features: int) -> list[float]:
    # 1. Инициализируем вектор нулями
    result = [0.0] * n_features
    
    # 2. Проходим по всем парам (название_признака, значение)
    for name, value in features.items():
        # Шаг 1: Вычисляем MD5 хэш
        digest = hashlib.md5(name.encode('utf-8')).hexdigest()
        
        # Шаг 2: Первые 8 hex-символов -> индекс корзины
        bucket_index = int(digest[:8], 16) % n_features
        
        # Шаг 3: Следующие 8 hex-символов -> знак (+1 или -1)
        sign_val = int(digest[8:16], 16)
        sign = 1 if sign_val % 2 == 0 else -1
        
        # Шаг 4: Прибавляем sign * value в соответствующую корзину
        result[bucket_index] += sign * float(value)
        
    return result