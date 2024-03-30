import argparse
import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

DATA_PATH = "data"
RAW_PATH = os.path.join(DATA_PATH, "raw")
PROCESSED_PATH = os.path.join(DATA_PATH, "processed")
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, raw_path=RAW_PATH):
    os.makedirs(raw_path, exist_ok=True)
    tgz_path = os.path.join(raw_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=raw_path)
    housing_tgz.close()


def load_housing_data(raw_path=RAW_PATH):
    csv_path = os.path.join(raw_path, "housing.csv")
    return pd.read_csv(csv_path)


def prepare_data_for_training(housing_data, processed_path):
    # Handle missing values
    imputer = SimpleImputer(strategy="median")
    housing_num = housing_data.drop("ocean_proximity", axis=1)  # Assuming 'ocean_proximity' is a categorical column
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)

    # Save processed data
    os.makedirs(processed_path, exist_ok=True)
    processed_file_path = os.path.join(processed_path, "housing_processed.csv")
    housing_tr.to_csv(processed_file_path, index=False)


def main(output_folder):
    fetch_housing_data(raw_path=RAW_PATH)
    housing_data = load_housing_data(raw_path=RAW_PATH)
    prepare_data_for_training(housing_data, processed_path=output_folder)
    print("Data ingestion completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and create training and validation datasets.")
    parser.add_argument("output_folder", help="Output folder/file path")
    args = parser.parse_args()
    main(args.output_folder)
