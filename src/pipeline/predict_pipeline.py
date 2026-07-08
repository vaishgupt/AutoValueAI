import os
import sys
import pandas as pd

from src.exception import error_message
from src.utils import load_object
from src.config import ModelTrainerConfig
from src.config import DataTransformationConfig


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:

            model_path = ModelTrainerConfig().trained_model_file_path
            preprocessor_path = DataTransformationConfig().preprocessor_path

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            data_scaled = preprocessor.transform(features)

            prediction = model.predict(data_scaled)


            return prediction

        except Exception as e:
            raise Exception(error_message(e, sys))
        



class CustomData:

    def __init__(
        self,
        name: str,
        year: int,
        km_driven: int,
        fuel: str,
        seller_type: str,
        transmission: str,
        owner: str
    ):

        self.name = name
        self.year = year
        self.km_driven = km_driven
        self.fuel = fuel
        self.seller_type = seller_type
        self.transmission = transmission
        self.owner = owner

    def get_data_as_dataframe(self):

        try:

            custom_data_input_dict = {
                "name": [self.name],
                "year": [self.year],
                "km_driven": [self.km_driven],
                "fuel": [self.fuel],
                "seller_type": [self.seller_type],
                "transmission": [self.transmission],
                "owner": [self.owner]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise Exception(error_message(e, sys))
        

if __name__ == "__main__":

    data = CustomData(
        name="Maruti Swift Dzire VDI",
        year=2018,
        km_driven=45000,
        fuel="Diesel",
        seller_type="Individual",
        transmission="Manual",
        owner="First Owner"
    )

    pred_df = data.get_data_as_dataframe()

    print("\nInput Data")
    print(pred_df)

    predict_pipeline = PredictPipeline()

    prediction = predict_pipeline.predict(pred_df)

    print("\nPredicted Selling Price:")
    print(f"₹ {prediction[0]:,.2f}")