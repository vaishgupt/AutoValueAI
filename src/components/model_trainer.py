import sys
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
)
from sklearn.neighbors import KNeighborsRegressor

from src.exception import error_message
from src.logger import logging
from src.utils import evaluate_models, save_object
from src.config import ModelTrainerConfig


class ModelTrainer:

    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):

        try:
            logging.info("Splitting training and testing arrays")

            # Split features and target
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            # Models
            models = {

                "Linear Regression": LinearRegression(),

                "Decision Tree": DecisionTreeRegressor(random_state=42),

                "Random Forest": RandomForestRegressor(random_state=42),

                "Gradient Boosting": GradientBoostingRegressor(random_state=42),

                "Extra Trees": ExtraTreesRegressor(random_state=42),

                "AdaBoost": AdaBoostRegressor(random_state=42),

                "KNN": KNeighborsRegressor()

            }

            # Hyperparameters
            params = {

                "Linear Regression": {},

                "Decision Tree": {
                    "criterion": ["squared_error", "absolute_error"],
                    "max_depth": [5, 10, 15]
                },

                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100, 200]
                },

                "Extra Trees": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20]
                },

                "AdaBoost": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [50, 100]
                },

                "KNN": {
                    "n_neighbors": [3, 5, 7]
                }

            }

            logging.info("Evaluating models...")

            model_report, best_model = evaluate_models(
                X_train,
                y_train,
                X_test,
                y_test,
                models,
                params
            )

            print("\nModel Performance")
            print("-" * 40)

            for model_name, score in model_report.items():
                print(f"{model_name:<20}: {score:.4f}")

            # Find best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            print("\nBest Model :", best_model_name)
            print(f"Best R² Score : {best_model_score:.4f}")

            

            # Save Best Model
            save_object(
                file_path=self.config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Best model saved successfully.")

            return model_report

        except Exception as e:
            raise Exception(error_message(e, sys))