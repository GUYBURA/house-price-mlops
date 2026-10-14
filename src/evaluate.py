import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, root_mean_squared_log_error

def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "rmse_log": root_mean_squared_error(np.log1p(y_true), np.log1p(y_pred)),
        "rmse_dollars": root_mean_squared_error(y_true, y_pred)
    }