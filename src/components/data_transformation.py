import os
import sys

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import error_message
from src.logger import logging
from src.utils import save_object


class DataTransformation:

    def get_data_transformer_object(self):
        """
        Creates preprocessing pipeline.
        """

        try:

            numeric_features = [
                "year",
                "km_driven"
            ]

            categorical_features = [
                "name",
                "fuel",
                "seller_type",
                "transmission",
                "owner"
            ]

            # Numeric Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehotencoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]
            )

            logging.info("Creating preprocessing object")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numeric_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise Exception(error_message(e, sys))

    def initiate_data_transformation(self, train_path, test_path):

        try:

            logging.info("Reading train and test data")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "selling_price"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logging.info("Getting preprocessing object")

            preprocessor = self.get_data_transformer_object()

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            os.makedirs("artifacts", exist_ok=True)

            save_object(
                file_path="artifacts/preprocessor.pkl",
                obj=preprocessor
            )

            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            logging.info("Data Transformation Completed Successfully")

            return train_arr, test_arr

        except Exception as e:
            raise Exception(error_message(e, sys))


if __name__ == "__main__":

    from src.components.data_ingestion import DataIngestion

    ingestion = DataIngestion()

    train_path, test_path = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()

    train_arr, test_arr = transformer.initiate_data_transformation(
        train_path,
        test_path
    )

    print("Train Shape:", train_arr.shape)
    print("Test Shape :", test_arr.shape)