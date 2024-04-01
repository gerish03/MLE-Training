import numpy as np

from HousePricePrediction.ingest_data import (StratifiedShuffleSplit_data,
                                              fetch_housing_data,
                                              load_housing_data,
                                              preprocessing_data)
from HousePricePrediction.score import score_model_mae, score_model_rmse
from HousePricePrediction.train import (grid_tune_random_forest,
                                        rand_tune_random_forest,
                                        train_decision_tree,
                                        train_linear_regression)

# Fetch and load data
fetch_housing_data()
housing = load_housing_data()
strat_train_set, strat_test_set = StratifiedShuffleSplit_data(housing)

X_train, y_train = preprocessing_data(strat_train_set)
X_test, y_test = preprocessing_data(strat_test_set)
LR_model = train_linear_regression(X_train, y_train)

DT_model = train_decision_tree(X_train, y_train)

rand_tune_RF_model = rand_tune_random_forest(X_train, y_train)
rand_cvres = rand_tune_RF_model.cv_results_

grid_tune_Tuned_RF_model = grid_tune_random_forest(X_train, y_train)
grid_tune_Tuned_RF_model.best_params_
grid_cvres = grid_tune_Tuned_RF_model.cv_results_

feature_importances = (
    grid_tune_Tuned_RF_model.best_estimator_.feature_importances_
)
sorted(zip(feature_importances, X_train.columns), reverse=True)

final_model = grid_tune_Tuned_RF_model.best_estimator_

print(
    "Linear Reggresion model RMSE score : ",
    score_model_rmse(LR_model, X_train, y_train),
)
print(
    "Linear Reggresion model MAE score : ",
    score_model_mae(LR_model, X_train, y_train),
)
print(
    "Decision Tree Regressor model RMSE score : ",
    score_model_rmse(DT_model, X_train, y_train),
)
print("Random Forest using RandomizedSearchCV model score : ")
for mean_score, params in zip(
    rand_cvres["mean_test_score"], rand_cvres["params"]
):
    print(np.sqrt(-mean_score), params)
print("Random Forest using GridSearchCV model score : ")
for mean_score, params in zip(
    grid_cvres["mean_test_score"], grid_cvres["params"]
):
    print(np.sqrt(-mean_score), params)
final_score = score_model_rmse(final_model, X_test, y_test)
print("Final model RMSE on the test set:", final_score)
