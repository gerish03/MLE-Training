import argparse
import logging
import os

from HousePricePrediction.ingest_data import (StratifiedShuffleSplit_data,
                                              fetch_housing_data,
                                              load_housing_data,
                                              preprocessing_data)


def setup_logger(log_level, log_to_file=True):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if log_to_file:
        file_handler = logging.FileHandler('ingest_data.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

def data_ingestion(raw_dataset_path, processed_dataset_path, log_level=logging.INFO, log_to_file=True):
    logger = setup_logger(log_level, log_to_file)
    logger.info("Fetching housing data...")
    fetch_housing_data(housing_path=raw_dataset_path)
    logger.info("Loading housing data...")
    housing_data = load_housing_data(raw_dataset_path)
    logger.info("Splitting housing data into train and test sets...")
    strat_train_set, strat_test_set = StratifiedShuffleSplit_data(housing_data)
    logger.info("Preprocessing training data...")
    X_train, y_train = preprocessing_data(strat_train_set)
    logger.info("Preprocessing testing data...")
    X_test, y_test = preprocessing_data(strat_test_set)

    os.makedirs(processed_dataset_path, exist_ok=True)
    train_data_path = os.path.join(processed_dataset_path, "X_train.csv")
    test_data_path = os.path.join(processed_dataset_path, "X_test.csv")
    train_labels_path = os.path.join(processed_dataset_path, "y_train.csv")
    test_labels_path = os.path.join(processed_dataset_path, "y_test.csv")

    logger.info("Saving preprocessed data to CSV files...")
    X_train.to_csv(train_data_path, index=False)
    X_test.to_csv(test_data_path, index=False)
    y_train.to_csv(train_labels_path, index=False)
    y_test.to_csv(test_labels_path, index=False)
    logger.info("Data ingestion completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess housing data")
    parser.add_argument("raw_dataset_path", type=str, nargs="?", default="data/raw", help="Input datasets folder (default: data/raw)")
    parser.add_argument("processed_dataset_path", type=str, nargs="?", default="data/processed", help="Processed Dataset folder (default: data/processed)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (default: INFO)")
    parser.add_argument("--log-path", type=str, help="Path to save log file")
    parser.add_argument("--no-console-log", action="store_false", help="Disable logging to console")
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_to_file = True if args.log_path else False
    data_ingestion(args.raw_dataset_path, args.processed_dataset_path, log_level=log_level, log_to_file=log_to_file)
