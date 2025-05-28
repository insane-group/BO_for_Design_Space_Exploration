import numpy as np

def calculate_r2_score(y_true, y_pred):
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def explained_variance(y_true, y_pred):
    var_true = np.var(y_true, ddof=1)
    var_res = np.var(y_true - y_pred, ddof=1)
    r2 = 1 - (var_res / var_true)
    return r2