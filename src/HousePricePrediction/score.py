import os

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def score_model_rmse(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return rmse


def score_model_mae(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    return mae


def save_metrics(model_name, score_type, score):
    metrics_dir = "artifacts/metrics/"
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    filename = os.path.join(metrics_dir, f"{model_name}_{score_type}.txt")

    with open(filename, "w") as f:
        f.write(str(score))


def RF_score(cvres, model):
    model_name = type(model).__name__
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        rmse_score = np.sqrt(-mean_score)
        save_metrics(model_name, "RMSE", rmse_score)
        print(rmse_score, params)
