import numpy as np
from scipy import stats

def descriptive_statistics(data: list | np.ndarray) -> dict:
    data = np.array(data)
    res = {}
    res['mean'] = data.mean()
    res['median'] = np.percentile(data, 50)
    res['mode'] = stats.mode(data, keepdims=False).mode
    res['variance'] = data.var()
    res['standard_deviation'] = data.std()
    res['25th_percentile'] = np.percentile(data, 25)
    res['50th_percentile'] = np.percentile(data, 50)
    res['75th_percentile'] = np.percentile(data, 75)
    res['interquartile_range'] = np.percentile(data, 75) - np.percentile(data, 25)
    return res