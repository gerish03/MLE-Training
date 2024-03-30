import os
import unittest
from unittest.mock import patch

from HousePricePrediction.score import RF_score, save_metrics


class TestScoreFunctions(unittest.TestCase):

    def setUp(self):
        self.model_name = "test_model"
        self.score_type = "RMSE"
        self.score = 0.0

    def test_save_metrics(self):
        save_metrics(self.model_name, self.score_type, self.score)
        filename = f"artifacts/metrics/{self.model_name}_{self.score_type}.txt"
        assert os.path.exists(filename), f"File {filename} was not created."

    @patch("builtins.print")
    def test_RF_score(self, mock_print):

        cvres = {
            "mean_test_score": [-100, -200],
            "params": [{"param1": 1}, {"param2": 2}],
        }
        mock_model = type("MockModel", (), {})
        RF_score(cvres, mock_model)
        filename = f"artifacts/metrics/{type(mock_model).__name__}_RMSE.txt"
        assert os.path.exists(filename), f"File {filename} was not created."
        assert (
            mock_print.call_count == 2
        ), "Print function was not called twice."


if __name__ == "__main__":
    unittest.main()
