from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def main():

    print("=" * 60)
    print("🚗 AutoValueAI Training Pipeline Started")
    print("=" * 60)

    print("\n[1/3] Running Data Ingestion...")
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    print("✅ Data Ingestion Completed")

    print("\n[2/3] Running Data Transformation...")
    transformation = DataTransformation()
    train_arr, test_arr = transformation.initiate_data_transformation(
        train_path,
        test_path
    )
    print("✅ Data Transformation Completed")

    print("\n[3/3] Running Model Training...")
    trainer = ModelTrainer()
    trainer.initiate_model_trainer(train_arr, test_arr)
    print("✅ Model Training Completed")

    print("\n" + "=" * 60)
    print("🎉 AutoValueAI Pipeline Completed Successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()