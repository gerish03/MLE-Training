import unittest
from unittest.mock import MagicMock

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from HousePricePrediction.score import (RF_score, score_model_mae,
                                        score_model_rmse)


class TestScoreFunctions(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X_test = np.random.rand(100, 10)
        self.y_test = np.random.rand(100)
        self.cvres = {"mean_test_score": np.random.rand(10), "params": ["params" + str(i) for i in range(10)]}
        self.model = MagicMock()
        self.model.predict.return_value = np.random.rand(100)

    def test_score_model_rmse(self):
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        rmse = np.sqrt(mse)
        self.assertIsInstance(rmse, float)

    def test_score_model_mae(self):
        predictions = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, predictions)
        self.assertIsInstance(mae, float)


    def test_RF_score(self):
        for mean_score, params in zip(self.cvres["mean_test_score"], self.cvres["params"]):
            self.assertIsInstance(mean_score, float)
            self.assertIsInstance(params, str)

if __name__ == '__main__':
    unittest.main()
