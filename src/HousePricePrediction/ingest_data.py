import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("data", "raw")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def StratifiedShuffleSplit_data(housing):
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

    return strat_train_set, strat_test_set


def preprocessing_data(housing_data):
    X = housing_data.drop("median_house_value", axis=1)
    y = housing_data["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")

    housing_num = X.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X_transformed = imputer.transform(housing_num)

    X_tr = pd.DataFrame(
        X_transformed, columns=housing_num.columns, index=X.index
    )

    X_tr["rooms_per_household"] = X_tr["total_rooms"] / X_tr["households"]
    X_tr["bedrooms_per_room"] = X_tr["total_bedrooms"] / X_tr["total_rooms"]
    X_tr["population_per_household"] = X_tr["population"] / X_tr["households"]

    housing_cat = X[["ocean_proximity"]]
    X_prepared = X_tr.join(pd.get_dummies(housing_cat, drop_first=True))
    return X_prepared, y

