import os
import unittest

import pandas as pd

from HousePricePrediction.ingest_data import (fetch_housing_data,
                                              load_housing_data,
                                              prepare_data_for_training)


class TestIngestData(unittest.TestCase):
    def setUp(self):
        self.raw_path = "data/raw"
        self.processed_path = "data/processed"

    def test_fetch_housing_data(self):
        try:
            # Test fetch_housing_data function
            fetch_housing_data(raw_path=self.raw_path)
            self.assertTrue(os.path.exists(os.path.join(self.raw_path, "housing.csv")))
        except Exception as e:
            self.fail(f"fetch_housing_data raised an exception: {e}")

    def test_load_housing_data(self):
        try:
            # Test load_housing_data function
            housing_data = load_housing_data(raw_path=self.raw_path)
            self.assertIsInstance(housing_data, pd.DataFrame)
        except Exception as e:
            self.fail(f"load_housing_data raised an exception: {e}")

    def test_prepare_data_for_training(self):
        try:
            # Load housing data
            housing_data = load_housing_data(raw_path=self.raw_path)
            # Test prepare_data_for_training function
            X_train, X_test, y_train, y_test = prepare_data_for_training(housing_data, processed_path=self.processed_path)

            self.assertIsInstance(X_train, pd.DataFrame)
            self.assertIsInstance(X_test, pd.DataFrame)
            self.assertIsInstance(y_train, pd.Series)
            self.assertIsInstance(y_test, pd.Series)
            self.assertTrue(os.path.exists(os.path.join(self.processed_path, "train_raw.csv")))
            self.assertTrue(os.path.exists(os.path.join(self.processed_path, "test_raw.csv")))
            self.assertTrue(os.path.exists(os.path.join(self.processed_path, "train_processed.csv")))
            self.assertTrue(os.path.exists(os.path.join(self.processed_path, "test_processed.csv")))
        except Exception as e:
            self.fail(f"prepare_data_for_training raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
