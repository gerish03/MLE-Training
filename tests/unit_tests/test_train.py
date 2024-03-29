import unittest

import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

from HousePricePrediction.train import (grid_tune_random_forest,
                                        rand_tune_random_forest,
                                        train_decision_tree,
                                        train_linear_regression)


class TestTrainFunctions(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.rand(100)

    def test_train_linear_regression(self):
        lin_reg = LinearRegression()
        lin_reg.fit(self.X_train, self.y_train)
        self.assertIsInstance(lin_reg, LinearRegression)


    def test_train_decision_tree(self):
        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(self.X_train, self.y_train)
        self.assertIsInstance(tree_reg, DecisionTreeRegressor)


    def test_rand_tune_random_forest(self):
        param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
        }

        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(
            forest_reg,
            param_distributions=param_distribs,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        rnd_search.fit(self.X_train, self.y_train)
        self.assertIsInstance(rnd_search, RandomizedSearchCV)


    def test_grid_tune_random_forest(self):
        param_grid = [
            {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
            {
                "bootstrap": [False],
                "n_estimators": [3, 10],
                "max_features": [2, 3, 4],
            },
        ]
        forest_reg = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            forest_reg,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )
        grid_search.fit(self.X_train, self.y_train)
        self.assertIsInstance(grid_search, GridSearchCV)


if __name__ == '__main__':
    unittest.main()
