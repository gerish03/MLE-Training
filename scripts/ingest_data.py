import argparse
import os

from HousePricePrediction.ingest_data import (StratifiedShuffleSplit_data,
                                              fetch_housing_data,
                                              load_housing_data,
                                              preprocessing_data)


def data_ingestion(raw_dataset_path, processed_dataset_path):
    print("Fetching housing data...")
    fetch_housing_data(housing_path=raw_dataset_path)
    print("Loading housing data...")
    housing_data = load_housing_data(raw_dataset_path)
    print("Splitting housing data into train and test sets...")
    strat_train_set, strat_test_set = StratifiedShuffleSplit_data(housing_data)
    print("Preprocessing training data...")
    X_train, y_train = preprocessing_data(strat_train_set)
    print("Preprocessing testing data...")
    X_test, y_test = preprocessing_data(strat_test_set)

    os.makedirs(processed_dataset_path, exist_ok=True)
    train_data_path = os.path.join(processed_dataset_path, "X_train.csv")
    test_data_path = os.path.join(processed_dataset_path, "X_test.csv")
    train_labels_path = os.path.join(processed_dataset_path, "y_train.csv")
    test_labels_path = os.path.join(processed_dataset_path, "y_test.csv")

    print("Saving preprocessed data to CSV files...")
    X_train.to_csv(train_data_path, index=False)
    X_test.to_csv(test_data_path, index=False)
    y_train.to_csv(train_labels_path, index=False)
    y_test.to_csv(test_labels_path, index=False)
    print("Data ingestion completed!")


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
    args = parser.parse_args()

    data_ingestion(args.raw_dataset_path, args.processed_dataset_path)
