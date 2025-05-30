import sys, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from healthapp.Liver.constant.training_pipeline import TARGET_COLUMN  # TARGET_COLUMN should be 'Dataset'
from healthapp.Liver.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from healthapp.exception.exception import HealthAppException
from healthapp.logging.logger import logging
from healthapp.Liver.entity.config_entity import DataTransformationConfig
from healthapp.Liver.utils.main_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise HealthAppException(e, sys)

    def read_data(self, file_path) -> pd.DataFrame:
        """Loads data and verifies its existence."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"❌ Data file not found: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise HealthAppException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Starting liver data transformation process...")
        try:
            # Read data from the validated training file (instead of a missing data_file_path in config)
            patients = self.read_data(self.data_validation_artifact.valid_train_file_path)
            logging.info(f"Data shape: {patients.shape}")

            # Convert Gender to numeric if the column exists
            if "Gender" in patients.columns:
                patients["Gender"] = patients["Gender"].apply(lambda x: 1 if x.strip().lower() == "male" else 0)
                logging.info("Converted Gender column to numeric.")

            # Fill missing values with constant 0.94
            patients = patients.fillna(0.94)

            # Define the feature columns and target (Assuming TARGET_COLUMN is 'Dataset')
            feature_columns = [
                'Total_Bilirubin', 'Direct_Bilirubin',
                'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                'Total_Protiens', 'Albumin',
                'Albumin_and_Globulin_Ratio'
            ]
            X = patients[feature_columns]
            y = patients[TARGET_COLUMN]
            logging.info(f"Extracted feature set X: {X.shape} and target y: {y.shape}")

            # Split the data into training and testing sets (70% train, 30% test, random_state=123)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=123
            )
            logging.info('Training set shape: X: {}, y: {}'.format(X_train.shape, y_train.shape))
            logging.info('Testing set shape: X: {}, y: {}'.format(X_test.shape, y_test.shape))

            # Combine features and target for saving as NumPy arrays
            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            # Save transformed arrays and a placeholder object (since no explicit pipeline is used)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, None)
            logging.info("Saved transformed training and testing data.")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact

        except Exception as e:
            raise HealthAppException(e, sys)