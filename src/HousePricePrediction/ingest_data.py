# -*- coding: utf-8 -*-
"""House Price Prediction - Data Ingestion Module

This module provides functions to fetch, load, and preprocess housing data for house price prediction.

Example
-------
To fetch and load housing data:

    >>> fetch_housing_data()
    >>> housing_data = load_housing_data()

To preprocess the loaded housing data:

    >>> X_train, X_test, y_train, y_test = preprocessing_data(housing_data)
"""

import logging
import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("data", "raw")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """Fetches housing data from a URL and extracts it into a specified directory.

    Parameters
    ----------
    housing_url : str, optional
        The URL to download the housing data from.
    housing_path : str, optional
        The directory where the housing data will be saved.

    Returns
    -------
    None
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """Loads housing data from a CSV file.

    Parameters
    ----------
    housing_path : str, optional
        The directory containing the housing data.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the loaded housing data.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def preprocessing_data(housing_data):
    """Preprocesses the loaded housing data for machine learning.

    Parameters
    ----------
    housing_data : pandas.DataFrame
        The DataFrame containing the loaded housing data.

    Returns
    -------
    tuple
        A tuple containing preprocessed feature matrices (X_train, X_test)
        and target vectors (y_train, y_test) for training and testing.
    """
    X = housing_data.drop("median_house_value", axis=1)
    y = housing_data["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")
    housing_num = X.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X_transformed = imputer.transform(housing_num)
    X_tr = pd.DataFrame(X_transformed, columns=housing_num.columns, index=X.index)

    X_tr["rooms_per_household"] = X_tr["total_rooms"] / X_tr["households"]
    X_tr["bedrooms_per_room"] = X_tr["total_bedrooms"] / X_tr["total_rooms"]
    X_tr["population_per_household"] = X_tr["population"] / X_tr["households"]

    housing_cat = X[["ocean_proximity"]]
    X_prepared = X_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(X_prepared, housing_data["income_cat"]):
        X_train = X_prepared.loc[train_index]
        X_test = X_prepared.loc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

    return X_train, X_test, y_train, y_test
