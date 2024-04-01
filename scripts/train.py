import argparse
import logging
import os

import pandas as pd
from joblib import dump

from HousePricePrediction.train import (grid_tune_random_forest,
                                        rand_tune_random_forest,
                                        train_decision_tree,
                                        train_linear_regression)

# Configure logging
logger = logging.getLogger("script")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Define console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def setup_logging(log_level, log_path, console_log):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % log_level)
    logger.setLevel(numeric_level)

    if log_path:
        log_directory, log_file = os.path.split(log_path)
        if not log_directory:
            log_directory = "."
        os.makedirs(log_directory, exist_ok=True)
        if not log_file:
            log_file = "logfile.log"
        log_file_path = os.path.join(log_directory, log_file)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not console_log:
        logger.removeHandler(console_handler)


def training_model(processed_dataset_path, ml_model_path):
    logger.info("Starting model training...")
    os.makedirs(ml_model_path, exist_ok=True)
    X_train_path = os.path.join(processed_dataset_path, "X_train.csv")
    X_train = pd.read_csv(X_train_path)

    y_train_path = os.path.join(processed_dataset_path, "y_train.csv")
    y_train = pd.read_csv(y_train_path).iloc[:, 0]

    logger.info("Training Linear Regression model...")
    LR_model = train_linear_regression(X_train, y_train)

    logger.info("Training Decision Tree model...")
    DT_model = train_decision_tree(X_train, y_train)

    logger.info("Training Random Forest model with randomized search...")
    rand_tune_RF_model = rand_tune_random_forest(X_train, y_train)

    logger.info("Training Random Forest model with grid search...")
    grid_tune_Tuned_RF_model = grid_tune_random_forest(X_train, y_train)
    grid_tune_Tuned_RF_model.best_params_

    feature_importances = (
        grid_tune_Tuned_RF_model.best_estimator_.feature_importances_
    )
    sorted(zip(feature_importances, X_train.columns), reverse=True)

    final_model = grid_tune_Tuned_RF_model.best_estimator_

    logger.info("Saving trained models...")
    dump(LR_model, os.path.join(ml_model_path, "LR_model.pkl"))
    dump(DT_model, os.path.join(ml_model_path, "DT_model.pkl"))
    dump(
        rand_tune_RF_model,
        os.path.join(ml_model_path, "rand_tune_RF_model.pkl"),
    )
    dump(
        grid_tune_Tuned_RF_model,
        os.path.join(ml_model_path, "grid_tune_Tuned_RF_model.pkl"),
    )
    dump(final_model, os.path.join(ml_model_path, "final_model.pkl"))

    logger.info("Training completed and models saved to %s", ml_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Making ML model and Storing")
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
        "--log-level",
        default="INFO",
        help="Specify log level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument("--log-path", default="", help="Specify log file path")
    parser.add_argument(
        "--no-console-log", action="store_true", help="Disable console logging"
    )
    args = parser.parse_args()

    setup_logging(args.log_level, args.log_path, not args.no_console_log)
    training_model(args.processed_dataset_path, args.ml_model_path)
