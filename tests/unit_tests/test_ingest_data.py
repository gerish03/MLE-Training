import unittest

import pandas as pd

from HousePricePrediction.ingest_data import (StratifiedShuffleSplit_data,
                                              fetch_housing_data,
                                              load_housing_data,
                                              preprocessing_data)


class TestIngestDataFunctions(unittest.TestCase):
    def setUp(self):
        data = {
            "longitude": [-122.23, -122.22, -121.56, -120.96, -119.22],
            "latitude": [37.88, 37.86, 37.70, 38.50, 34.41],
            "housing_median_age": [41, 21, 32, 14, 20],
            "total_rooms": [880, 7099, 1467, 4571, 2814],
            "total_bedrooms": [129, 1106, 190, 655, 406],
            "population": [322, 2401, 496, 1473, 1139],
            "households": [126, 1138, 177, 753, 391],
            "median_income": [8.3252, 8.3014, 7.2574, 6.2298, 2.6512],
            "median_house_value": [452600, 358500, 352100, 341300, 342200],
            "ocean_proximity": [
                "NEAR BAY",
                "NEAR BAY",
                "INLAND",
                "NEAR OCEAN",
                "<1H OCEAN",
            ],
        }

        self.housing_df = pd.DataFrame(data)

    def test_fetch_housing_data(self):
        try:
            fetch_housing_data()
        except Exception as e:
            self.assertFalse, (
                f"Error: {e}. "
                "Not able to access fetch_housing_data function"
            )

    def test_load_housing_data(self):
        try:
            housing = load_housing_data()
        except Exception as e:
            self.assertTrue(housing is not None), (
                f"Error: {e}" "Could not able load the data to housing.csv"
            )

    def test_StratifiedShuffleSplit_data(self):
        try:
            train_set, test_set = StratifiedShuffleSplit_data(self.housing_df)
            assert isinstance(train_set, pd.DataFrame) and isinstance(
                test_set, pd.DataFrame
            )
        except Exception as e:
            self.assertFalse, (
                f"Error: {e}"
                "Could not able split the data via StratifiedShuffleSplit_data"
            )

    def test_prepare_data_for_training(self):
        try:
            X_train_prepared, y_train = preprocessing_data(self.housing_df)
            assert (X_train_prepared is not None) and (y_train is not None)
        except Exception as e:
            self.assertFalse, (f"Error: {e}" "Preprocessed data is None.")


if __name__ == "__main__":
    unittest.main()
