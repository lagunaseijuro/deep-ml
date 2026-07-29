import math

def normal_pdf(x: float, mean: float, std: float) -> float:
    # 1. Считаем коэффициент перед экспонентой
    coefficient = 1 / (std * math.sqrt(2 * math.pi))
    
    # 2. Считаем показатель экспоненты
    exponent = math.exp(-0.5 * ((x - mean) / std) ** 2)
    
    # 3. Перемножаем и округляем до 5 знаков
    return round(coefficient * exponent, 5)