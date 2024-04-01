import unittest

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from HousePricePrediction.score import score_model_mae, score_model_rmse


class TestModelScoring(unittest.TestCase):
    def setUp(self):
        self.X_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.y_test = np.array([10, 20, 30])

    def test_score_model_rmse(self):
        try:

            class MockModel:
                def predict(self, X):
                    return np.array([11, 22, 33])

            model = MockModel()
            rmse = score_model_rmse(model, self.X_test, self.y_test)
            expected_rmse = np.sqrt(
                mean_squared_error(self.y_test, [11, 22, 33])
            )
            self.assertAlmostEqual(rmse, expected_rmse)
        except Exception as e:
            self.fail(f"Unexpected error occurred during RMSE scoring: {e}")

    def test_score_model_mae(self):
        try:

            class MockModel:
                def predict(self, X):
                    return np.array([11, 22, 33])

            model = MockModel()
            mae = score_model_mae(model, self.X_test, self.y_test)
            expected_mae = mean_absolute_error(self.y_test, [11, 22, 33])
            self.assertAlmostEqual(mae, expected_mae)
        except Exception as e:
            self.fail(f"Unexpected error occurred during MAE scoring: {e}")


if __name__ == "__main__":
    unittest.main()
