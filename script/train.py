import argparse
import os

import joblib
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from HousePricePrediction.train import (grid_tune_random_forest,
                                        rand_tune_random_forest,
                                        train_decision_tree,
                                        train_linear_regression)


def load_data():
    # Generate sample data for demonstration
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    return X, y

def train_models(X_train, y_train, output_folder):
    # Train models
    models = {
        "lin_reg": train_linear_regression(X_train, y_train),
        "decision_tree": train_decision_tree(X_train, y_train),
        "rand_forest_grid_search": grid_tune_random_forest(X_train, y_train),
        "rand_forest_random_search": rand_tune_random_forest(X_train, y_train)
    }


    os.makedirs(output_folder, exist_ok=True)
    for model_name, model in models.items():
        model_path = os.path.join(output_folder, f"{model_name}_model.pkl")
        joblib.dump(model, model_path)
        print(f"Trained model '{model_name}' saved to: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model(s).")
    parser.add_argument("dataset_folder", help="Folder containing the dataset")
    parser.add_argument("output_folder", help="Folder for saving trained models")
    args = parser.parse_args()
    X, y = load_data()
    train_models(X, y, args.output_folder)
