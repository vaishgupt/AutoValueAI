import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import error_message
from src.logger import logging


class DataIngestion:

    def initiate_data_ingestion(self):

        try:
            logging.info("Reading Dataset")

            df = pd.read_csv("data/CAR DETAILS FROM CAR DEKHO.csv")

            os.makedirs("artifacts", exist_ok=True)

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            train_path = os.path.join("artifacts", "train.csv")
            test_path = os.path.join("artifacts", "test.csv")

            train_set.to_csv(train_path, index=False)
            test_set.to_csv(test_path, index=False)

            logging.info("Train Test Split Completed")

            return train_path, test_path

        except Exception as e:
            raise Exception(error_message(e, sys))


if __name__ == "__main__":

    obj = DataIngestion()

    train_path, test_path = obj.initiate_data_ingestion()

    print(train_path)
    print(test_path)