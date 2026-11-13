from collections import Counter

def balance_undersample(data: list) -> list:
    """
    Undersample the majority classes so all classes have the same number of
    samples equal to the minority class count.

    data: list of (sample, label) tuples
    Returns: list of (sample, label) tuples, order-preserving
    """
    if not data:
        return []
        
    min_class_freq =  min(Counter([label for sample, label in data]).values())

    result = []
    curr_freq = {}
    for sample, label in data:
        if curr_freq.get(label, 0) < min_class_freq:
            result.append((sample, label))
            curr_freq[label] = curr_freq.get(label, 0) + 1
        
    return result
