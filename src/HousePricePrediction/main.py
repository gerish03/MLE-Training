from ingest_data import (fetch_housing_data, load_housing_data,
                         prepare_data_for_training)
from score import score_model
from train import (train_decision_tree, train_linear_regression,
                   train_random_forest, tune_random_forest)

# Fetch and load data
fetch_housing_data()
housing = load_housing_data()

X_train, X_test, y_train, y_test = prepare_data_for_training(housing)

LR_model = train_linear_regression(X_train, y_train)
DT_model = train_decision_tree(X_train, y_train)
RF_model = train_random_forest(X_train, y_train)
Tuned_RF_model = tune_random_forest(X_train, y_train)

score = score_model(Tuned_RF_model, X_test, y_test)
print("Model Score:", score)
