import sys

from src.exception import CustomException
from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion


class TrainingPipeline:
    def __init__(self):
        pass

    def train(self):
        try:
            di_obj = DataIngestion()
            train_data_path, test_data_path = di_obj.initiate_data_ingestion()

            dt_obj = DataTransformation()
            train_arr, test_arr, _ = dt_obj.initiate_data_transformation(
                train_data_path, test_data_path)

            print("Before Training")

            mt_obj = ModelTrainer()
            mt_obj.initiate_model_trainer(
                train_array=train_arr, test_array=test_arr)

            print("After Training")

        except Exception as e:
            raise CustomException(e, sys)
