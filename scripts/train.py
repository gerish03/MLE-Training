import argparse
import os

import pandas as pd
from joblib import dump

from HousePricePrediction.train import (grid_tune_random_forest,
                                        rand_tune_random_forest,
                                        train_decision_tree,
                                        train_linear_regression)


def training_model(processed_dataset_path, ml_model_path):
    print("Creating ML models...")
    os.makedirs(ml_model_path, exist_ok=True)
    print(f"Processed dataset path: {processed_dataset_path}")
    print(f"ML model path: {ml_model_path}")

    X_train_path = os.path.join(processed_dataset_path, "X_train.csv")
    print(f"Loading X_train data from: {X_train_path}")
    X_train = pd.read_csv(X_train_path)

    y_train_path = os.path.join(processed_dataset_path, "y_train.csv")
    print(f"Loading y_train data from: {y_train_path}")
    y_train = pd.read_csv(y_train_path).iloc[:, 0]

    print("Training Linear Regression model...")
    LR_model = train_linear_regression(X_train, y_train)

    print("Training Decision Tree model...")
    DT_model = train_decision_tree(X_train, y_train)

    print("Training Randomized Search CV Random Forest model...")
    rand_tune_RF_model = rand_tune_random_forest(X_train, y_train)

    print("Training Grid Search CV Random Forest model...")
    grid_tune_Tuned_RF_model = grid_tune_random_forest(X_train, y_train)
    grid_tune_Tuned_RF_model.best_params_

    feature_importances = (
        grid_tune_Tuned_RF_model.best_estimator_.feature_importances_
    )
    print("Feature Importances:")
    print(sorted(zip(feature_importances, X_train.columns), reverse=True))

    final_model = grid_tune_Tuned_RF_model.best_estimator_

    print("Saving trained models...")
    dump(LR_model, os.path.join(ml_model_path, "LR_model.pkl"))
    dump(DT_model, os.path.join(ml_model_path, "DT_model.pkl"))
    dump(
        rand_tune_RF_model,
        os.path.join(ml_model_path, "rand_tune_RF_model.pkl"),
    )
    dump(
        grid_tune_Tuned_RF_model,
        os.path.join(ml_model_path, "grid_tune_Tuned_RF_model.pkl"),
    )
    dump(final_model, os.path.join(ml_model_path, "final_model.pkl"))

    print("Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Making ML model and Storing")
    parser.add_argument(
        "processed_dataset_path",
        type=str,
        nargs="?",
        default="data/processed",
        help="Processed Dataset folder (default: data/processed)",
    )

    parser.add_argument(
        "ml_model_path",
        type=str,
        nargs="?",
        default="artifacts/models",
        help="ML Models folder (default: artifacts/models)",
    )
    args = parser.parse_args()

    training_model(args.processed_dataset_path, args.ml_model_path)
