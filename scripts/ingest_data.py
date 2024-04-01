import argparse
import logging
import os

from HousePricePrediction.ingest_data import (StratifiedShuffleSplit_data,
                                              fetch_housing_data,
                                              load_housing_data,
                                              preprocessing_data)
from HousePricePrediction.logger import setup_logging


def data_ingestion(raw_dataset_path, processed_dataset_path, logger):
    logger.info("Starting data ingestion process...")
    fetch_housing_data(housing_path=raw_dataset_path)
    logger.info("Housing data fetched successfully.")
    housing_data = load_housing_data(raw_dataset_path)
    logger.info("Housing data loaded successfully.")
    strat_train_set, strat_test_set = StratifiedShuffleSplit_data(housing_data)
    logger.info("Data split into train and test sets.")
    X_train, y_train = preprocessing_data(strat_train_set)
    logger.info("Preprocessing of training data completed.")
    X_test, y_test = preprocessing_data(strat_test_set)
    logger.info("Preprocessing of test data completed.")

    os.makedirs(processed_dataset_path, exist_ok=True)
    train_data_path = os.path.join(processed_dataset_path, "X_train.csv")
    test_data_path = os.path.join(processed_dataset_path, "X_test.csv")
    train_labels_path = os.path.join(processed_dataset_path, "y_train.csv")
    test_labels_path = os.path.join(processed_dataset_path, "y_test.csv")

    X_train.to_csv(train_data_path, index=False)
    X_test.to_csv(test_data_path, index=False)
    y_train.to_csv(train_labels_path, index=False)
    y_test.to_csv(test_labels_path, index=False)
    logger.info("Data saved to processed dataset folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and preprocess housing data"
    )
    parser.add_argument(
        "raw_dataset_path",
        type=str,
        nargs="?",
        default="data/raw",
        help="Input datasets folder (default: data/raw)",
    )
    parser.add_argument(
        "processed_dataset_path",
        type=str,
        nargs="?",
        default="data/processed",
        help="Processed Dataset folder (default: data/processed)",
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

    logger = setup_logging("ingest_data", args.log_level, args.log_path, not args.no_console_log)
    data_ingestion(args.raw_dataset_path, args.processed_dataset_path, logger)