from dataclasses import dataclass
import os

@dataclass
class DataIngestionConfig:

    train_data_path: str = "artifacts/train.csv"

    test_data_path: str = "artifacts/test.csv"

    raw_data_path: str = "artifacts/raw.csv"


@dataclass
class DataTransformationConfig:

    preprocessor_path: str = "artifacts/preprocessor.pkl"


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )