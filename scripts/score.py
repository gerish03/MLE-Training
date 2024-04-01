import argparse
import logging
import os

import numpy as np
import pandas as pd
from joblib import load

from HousePricePrediction.logger import setup_logging
from HousePricePrediction.score import score_model_mae, score_model_rmse

logger = setup_logging(__name__, logging.INFO, "score.log", console_log=True)

def scoring(processed_dataset_path, ml_model_path, scoring_path):
    logger.info("Starting model scoring...")
    os.makedirs(scoring_path, exist_ok=True)

    X_train = pd.read_csv(os.path.join(processed_dataset_path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dataset_path, "X_test.csv"))
    y_train = pd.read_csv(
        os.path.join(processed_dataset_path, "y_train.csv")
    ).iloc[:, 0]
    y_test = pd.read_csv(
        os.path.join(processed_dataset_path, "y_test.csv")
    ).iloc[:, 0]

    LR_model = load(os.path.join(ml_model_path, "LR_model.pkl"))
    DT_model = load(os.path.join(ml_model_path, "DT_model.pkl"))
    rand_tune_RF_model = load(
        os.path.join(ml_model_path, "rand_tune_RF_model.pkl")
    )
    rand_cvres = rand_tune_RF_model.cv_results_
    grid_tune_Tuned_RF_model = load(
        os.path.join(ml_model_path, "grid_tune_Tuned_RF_model.pkl")
    )
    grid_cvres = grid_tune_Tuned_RF_model.cv_results_
    final_model = load(os.path.join(ml_model_path, "final_model.pkl"))

    with open(os.path.join(scoring_path, "LR_model_scores.txt"), "w") as f:
        f.write(
            "Linear Regression model RMSE score: {}\n".format(
                score_model_rmse(LR_model, X_train, y_train)
            )
        )
        f.write(
            "Linear Regression model MAE score: {}\n".format(
                score_model_mae(LR_model, X_train, y_train)
            )
        )

    with open(os.path.join(scoring_path, "DT_model_score.txt"), "w") as f:
        f.write(
            "Decision Tree Regressor model RMSE score: {}\n".format(
                score_model_rmse(DT_model, X_train, y_train)
            )
        )

    with open(
        os.path.join(scoring_path, "rand_tune_RF_model_scores.txt"), "w"
    ) as f:
        f.write("Random Forest using RandomizedSearchCV model score: \n")
        for mean_score, params in zip(
            rand_cvres["mean_test_score"], rand_cvres["params"]
        ):
            f.write("{} {}\n".format(np.sqrt(-mean_score), params))

    with open(
        os.path.join(scoring_path, "grid_tune_Tuned_RF_model_scores.txt"), "a"
    ) as f:
        f.write("Random Forest using GridSearchCV model score: \n")
        for mean_score, params in zip(
            grid_cvres["mean_test_score"], grid_cvres["params"]
        ):
            f.write("{} {}\n".format(np.sqrt(-mean_score), params))

    with open(os.path.join(scoring_path, "final_model_score.txt"), "w") as f:
        final_score = score_model_rmse(final_model, X_test, y_test)
        f.write("Final model RMSE on the test set: {}\n".format(final_score))

    dt_rmse_score = score_model_rmse(DT_model, X_train, y_train)
    logger.info("Decision Tree Regressor model RMSE score: %f", dt_rmse_score)

    # Score Random Forest models
    logger.info("Scoring Random Forest models...")
    rand_cvres = rand_tune_RF_model.cv_results_
    for mean_score, params in zip(rand_cvres["mean_test_score"], rand_cvres["params"]):
        logger.info("Random Forest using RandomizedSearchCV model RMSE score: %f", np.sqrt(-mean_score))
        logger.info("Parameters: %s", params)

    grid_cvres = grid_tune_Tuned_RF_model.cv_results_
    for mean_score, params in zip(grid_cvres["mean_test_score"], grid_cvres["params"]):
        logger.info("Random Forest using GridSearchCV model RMSE score: %f", np.sqrt(-mean_score))
        logger.info("Parameters: %s", params)

    # Score final model
    final_score = score_model_rmse(final_model, X_test, y_test)
    logger.info("Final model RMSE on the test set: %f", final_score)

    logger.info("Scoring completed and results saved to %s", scoring_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scoring of Model and storing them in metrics"
    )
    parser.add_argument(
        "processed_dataset_path",
        type=str,
        nargs="?",
        default="data/processed",
        help="Processed Dataset folder (default: data/processed)",
    )
    parser.add_argument(
        "ml_model_path",
        type=str,
        nargs="?",
        default="artifacts/models",
        help="ML Models folder (default: artifacts/models)",
    )
    parser.add_argument(
        "scoring_path",
        type=str,
        nargs="?",
        default="artifacts/metrics",
        help="Model Metrics folder (default: artifacts/metrics)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Specify log level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument("--log-path", default="", help="Specify log file path")
    parser.add_argument(
        "--no-console-log", action="store_true", help="Disable console logging"
    )
    args = parser.parse_args()

    logger = setup_logging("score", args.log_level, args.log_path, not args.no_console_log)
    scoring(args.processed_dataset_path, args.ml_model_path, args.scoring_path)


