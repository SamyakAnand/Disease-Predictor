import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Use Liver-specific constants, entities, etc.
from healthapp.Liver.constant.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from healthapp.Liver.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from healthapp.exception.exception import HealthAppException
from healthapp.logging.logger import logging
from healthapp.Liver.entity.config_entity import DataTransformationConfig
from healthapp.Liver.utils.main_utils.utils import save_numpy_array_data, save_object

# Define the feature columns for Liver data transformation.
# These keys should match those used in your liver results template.
FEATURE_COLS = [
    'Total_Bilirubin', 
    'Direct_Bilirubin', 
    'Alkaline_Phosphotase', 
    'Alamine_Aminotransferase', 
    'Total_Protiens', 
    'Albumin', 
    'Albumin_and_Globulin_Ratio'
]

class DataTransformation:
    def __init__(self, 
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise HealthAppException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Loads data from the CSV file and immediately filters the DataFrame to keep only
        the required columns: FEATURE_COLS + [TARGET_COLUMN].
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"❌ Data file not found: {file_path}")
            df = pd.read_csv(file_path)
            required_cols = FEATURE_COLS + [TARGET_COLUMN]
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"❌ Missing columns from CSV: {missing_cols}")
            return df[required_cols]
        except Exception as e:
            raise HealthAppException(e, sys)

    def get_data_transformer_obj(self) -> Pipeline:
        """
        Initializes a preprocessing pipeline consisting of KNNImputer,
        StandardScaler, and MinMaxScaler for our selected liver features.
        """
        logging.info("Initializing preprocessing pipeline for liver data transformation.")
        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            scaler = StandardScaler()
            normalizer = MinMaxScaler()
            processor = Pipeline([
                ("imputer", imputer),      # Fill missing values
                ("scaler", scaler),        # Standardize data
                ("normalizer", normalizer) # Normalize data
            ])
            return processor
        except Exception as e:
            raise HealthAppException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Loads the training and testing data (filtered to FEATURE_COLS + TARGET_COLUMN),
        applies the transformation pipeline, and saves the transformed arrays as well as
        the preprocessor object.
        """
        logging.info("Starting liver data transformation process...")
        try:
            # Narrow down the CSV to only the required columns.
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            logging.info(f"Train Data Shape (after filtering): {train_df.shape}, "
                         f"Test Data Shape (after filtering): {test_df.shape}")

            if train_df.empty or test_df.empty:
                raise ValueError("❌ Train or Test dataset is empty. Check data ingestion.")

            # Separate features and target.
            input_feature_train_df = train_df[FEATURE_COLS]
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df[FEATURE_COLS]
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Fit the transformation pipeline on the training features.
            preprocessor = self.get_data_transformer_obj()
            preprocessor_obj = preprocessor.fit(input_feature_train_df)

            transformed_train_features = preprocessor_obj.transform(input_feature_train_df)
            transformed_test_features = preprocessor_obj.transform(input_feature_test_df)

            # Combine transformed features with the target.
            train_arr = np.c_[transformed_train_features, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_test_features, np.array(target_feature_test_df)]

            # Save transformed arrays.
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )
            # Save the preprocessor object.
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor_obj
            )
            # Also save the preprocessor to the designated liver models folder.
            save_object("healthapp/Liver/final_models/liver_preprocessor.pkl", preprocessor_obj)

            logging.info("Saved transformed training and testing data, and the preprocessor object.")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact

        except Exception as e:
            raise HealthAppException(e, sys)