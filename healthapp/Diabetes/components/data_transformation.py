import sys, os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from healthapp.Diabetes.constant.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from healthapp.Diabetes.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from healthapp.exception.exception import HealthAppException
from healthapp.logging.logger import logging
from healthapp.Diabetes.entity.config_entity import DataTransformationConfig
from healthapp.utils.main_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise HealthAppException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """Loads data and verifies its existence."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"❌ Data file not found: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise HealthAppException(e, sys)

    def get_data_transformer_obj(self) -> Pipeline:
        """Initializes KNNImputer + StandardScaler + MinMaxScaler pipeline."""
        logging.info("Initializing preprocessing pipeline for numerical data transformation.")
        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            scaler = StandardScaler()
            normalizer = MinMaxScaler()

            processor = Pipeline([
                ("imputer", imputer),      # Handles missing values
                ("scaler", scaler),        # Standardizes data
                ("normalizer", normalizer) # Normalizes data
            ])
            return processor
        except Exception as e:
            raise HealthAppException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Starting numerical data transformation process...")
        try:
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            logging.info(f"Train Data Shape: {train_df.shape}, Test Data Shape: {test_df.shape}")

            if train_df.empty or test_df.empty:
                raise ValueError("❌ Train or Test dataset is empty. Check data ingestion.")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            preprocessor = self.get_data_transformer_obj()
            preprocessor_obj = preprocessor.fit(input_feature_train_df)

            transformed_train_features = preprocessor_obj.transform(input_feature_train_df)
            transformed_test_features = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[transformed_train_features, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_test_features, np.array(target_feature_test_df)]

            # Save transformed data and preprocessing object
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_obj)

            save_object("healthapp/Diabetes/final_models/diabetes_preprocessor.pkl", preprocessor_obj)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact

        except Exception as e:
            raise HealthAppException(e, sys)