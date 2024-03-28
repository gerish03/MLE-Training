from HousePricePrediction.ingest_data import (fetch_housing_data,
                                              load_housing_data,
                                              prepare_data_for_training)
from HousePricePrediction.score import (RF_score, score_model_mae,
                                        score_model_rmse)
from HousePricePrediction.train import (grid_tune_random_forest,
                                        rand_tune_random_forest,
                                        train_decision_tree,
                                        train_linear_regression)

# Fetch and load data
fetch_housing_data()
housing = load_housing_data()

X_train, X_test, y_train, y_test = prepare_data_for_training(housing)

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
    "Decision Tree Regressor model score : ",
    score_model_rmse(DT_model, X_train, y_train),
)
print("Random Forest using RandomizedSearchCV model score : ")
RF_score(rand_cvres)

print("Random Forest using GridSearchCV model score : ")
RF_score(grid_cvres)

final_score = score_model_rmse(final_model, X_test, y_test)
print("Final model RMSE on the test set:", final_score)
