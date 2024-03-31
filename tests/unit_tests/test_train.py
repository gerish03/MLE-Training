import unittest

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
from sklearn.tree import DecisionTreeRegressor

from HousePricePrediction.train import (grid_tune_random_forest,
                                        rand_tune_random_forest,
                                        train_decision_tree,
                                        train_linear_regression)


class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_regression(
            n_samples=100, n_features=5, noise=0.1, random_state=42
        )
        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        )

    def test_train_linear_regression(self):
        try:
            model = train_linear_regression(self.X_train, self.y_train)
            self.assertIsInstance(
                model,
                LinearRegression,
                "The trained model is not an instance of LinearRegression",
            )
            self.assertTrue(
                np.any(model.coef_), "The trained model coefficients are empty"
            )
            y_pred = model.predict(self.X_test)
            self.assertIsNotNone(
                y_pred, "Failed to make predictions using the trained model"
            )
        except Exception as e:
            self.fail(f"Error occurred during linear regression training: {e}")

    def test_train_decision_tree(self):
        try:
            model = train_decision_tree(self.X_train, self.y_train)
            self.assertIsInstance(
                model,
                DecisionTreeRegressor,
                "The model is not an instance of DecisionTreeRegressor",
            )
            self.assertTrue(
                model.tree_.node_count > 1,
                "The decision tree model has an insufficient number of nodes",
            )
            y_pred = model.predict(self.X_test)
            self.assertIsNotNone(
                y_pred, "Failed to make predictions using the trained model"
            )
        except Exception as e:
            self.fail(
                f"Unexpected error occurred during decision tree training: {e}"
            )

    def test_rand_tune_random_forest(self):
        try:
            model = rand_tune_random_forest(self.X_train, self.y_train)
            self.assertIsInstance(
                model,
                RandomizedSearchCV,
                "The model is not an instance of RandomizedSearchCV",
            )
            self.assertTrue(
                hasattr(model, "best_estimator_"),
                "Randomized search did not find the best estimator",
            )
            y_pred = model.predict(self.X_test)
            self.assertIsNotNone(
                y_pred, "Failed to make predictions using the tuned model"
            )
        except Exception as e:
            self.fail(
                f"Unexpected error occurred during random forest tuning: {e}"
            )

    def test_grid_tune_random_forest(self):
        try:
            model = grid_tune_random_forest(self.X_train, self.y_train)
            self.assertIsInstance(
                model,
                GridSearchCV,
                "The model is not an instance of GridSearchCV",
            )
            self.assertTrue(
                hasattr(model, "best_estimator_"),
                "Grid search did not find the best estimator",
            )
            y_pred = model.predict(self.X_test)
            self.assertIsNotNone(
                y_pred, "Failed to make predictions using the tuned model"
            )
        except Exception as e:
            self.fail(
                f"Unexpected error occurred during random forest tuning: {e}"
            )


if __name__ == "__main__":
    unittest.main()
