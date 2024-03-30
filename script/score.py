import argparse
import os

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Score the model(s) on a dataset.')
    parser.add_argument('model_folder', type=str, help='Path to the folder containing the model(s).')
    parser.add_argument('dataset_folder', type=str, help='Path to the folder containing the dataset.')
    parser.add_argument('--output_folder', type=str, default='artifacts/metrics/',
                        help='Path to the folder to save the output metrics. Default is "artifacts/metrics/".')
    return parser.parse_args()


def score_model_rmse(model, X_test, y_test):
    """Compute RMSE for model predictions."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return rmse


def score_model_mae(model, X_test, y_test):
    """Compute MAE for model predictions."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    return mae


def save_metrics(model_name, score_type, score, output_folder):
    """Save computed metrics to a text file."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = os.path.join(output_folder, f"{model_name}_{score_type}.txt")

    with open(filename, "w") as f:
        f.write(str(score))


def RF_score(cvres, model, output_folder):
    """Score Random Forest model using cross-validation results."""
    model_name = type(model).__name__
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        rmse_score = np.sqrt(-mean_score)
        save_metrics(model_name, "RMSE", rmse_score, output_folder)
        print(rmse_score, params)


def main():
    args = parse_arguments()

if __name__ == "__main__":
    main()
