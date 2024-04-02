import logging
import os

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from HousePricePrediction.logger import setup_logging

logger = logging.getLogger(__name__)

logger = setup_logging(__name__, logging.INFO, "logs/Script_output.log", console_log=True)
logger.info("Executing score module...")


def score_model_rmse(model, X_test, y_test):
    """Calculate the Root Mean Squared Error (RMSE) for a regression model.

    Parameters
    ----------
    model : estimator
        The trained regression model.
    X_test : array-like of shape (n_samples, n_features)
        The test input samples.
    y_test : array-like of shape (n_samples,)
        The true target values.

    Returns
    -------
    float
        The calculated RMSE value.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return rmse


def score_model_mae(model, X_test, y_test):
    """Calculate the Mean Absolute Error (MAE) for a regression model.

    Parameters
    ----------
    model : estimator
        The trained regression model.
    X_test : array-like of shape (n_samples, n_features)
        The test input samples.
    y_test : array-like of shape (n_samples,)
        The true target values.

    Returns
    -------
    float
        The calculated MAE value.
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    return mae
