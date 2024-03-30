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


def prepare_data_for_training(housing, processed_path=PROCESSED_PATH):
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    os.makedirs(processed_path, exist_ok=True)

    train_raw_path = os.path.join(processed_path, "train_raw.csv")
    strat_train_set.to_csv(train_raw_path, index=False)
    test_raw_path = os.path.join(processed_path, "test_raw.csv")
    strat_test_set.to_csv(test_raw_path, index=False)

    X_train = strat_train_set.drop("median_house_value", axis=1)
    y_train = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")
    housing_num = X_train.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=X_train.index
    )

    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = X_train[["ocean_proximity"]]
    X_train_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
    )

    train_processed_path = os.path.join(processed_path, "train_processed.csv")
    X_train_prepared.to_csv(train_processed_path, index=False)

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(
        pd.get_dummies(X_test_cat, drop_first=True)
    )

    test_processed_path = os.path.join(processed_path, "test_processed.csv")
    X_test_prepared.to_csv(test_processed_path, index=False)

    return X_train_prepared, X_test_prepared, y_train, y_test
