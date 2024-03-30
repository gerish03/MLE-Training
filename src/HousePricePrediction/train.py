from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor


def train_linear_regression(X_train, y_train):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    return lin_reg


def train_decision_tree(X_train, y_train):
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(X_train, y_train)
    return tree_reg


def rand_tune_random_forest(X_train, y_train):
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
    rnd_search.fit(X_train, y_train)
    return rnd_search


def grid_tune_random_forest(X_train, y_train):
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
    grid_search.fit(X_train, y_train)
    return grid_search
