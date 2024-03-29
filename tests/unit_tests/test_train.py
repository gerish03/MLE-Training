import unittest

import numpy as np
from sklearn.datasets import make_regression

from HousePricePrediction.train import (grid_tune_random_forest,
                                        rand_tune_random_forest,
                                        train_decision_tree,
                                        train_linear_regression)


class TestTrainFunctions(unittest.TestCase):

    def setUp(self):
        # Generate sample data for testing
        self.X_train, self.y_train = make_regression(
            n_samples=100, n_features=10, noise=0.1, random_state=42
        )

    def test_train_linear_regression(self):
        try:
            lin_reg_model = train_linear_regression(self.X_train, self.y_train)
            self.assertIsNotNone(lin_reg_model)
            # Add more specific tests if needed
        except Exception as e:
            self.fail(
                f"train_linear_regression raised an unexpected exception: {e}"
            )

    def test_train_decision_tree(self):
        try:
            tree_reg_model = train_decision_tree(self.X_train, self.y_train)
            self.assertIsNotNone(tree_reg_model)
            # Add more specific tests if needed
        except Exception as e:
            self.fail(
                f"train_decision_tree raised an unexpected exception: {e}"
            )

    def test_rand_tune_random_forest(self):
        try:
            rnd_search_model = rand_tune_random_forest(
                self.X_train, self.y_train
            )
            self.assertIsNotNone(rnd_search_model)
            # Add more specific tests if needed
        except Exception as e:
            self.fail(
                f"rand_tune_random_forest raised an unexpected exception: {e}"
            )

    def test_grid_tune_random_forest(self):
        try:
            grid_search_model = grid_tune_random_forest(
                self.X_train, self.y_train
            )
            self.assertIsNotNone(grid_search_model)
            # Add more specific tests if needed
        except Exception as e:
            self.fail(
                f"grid_tune_random_forest raised an unexpected exception: {e}"
            )


if __name__ == "__main__":
    unittest.main()
