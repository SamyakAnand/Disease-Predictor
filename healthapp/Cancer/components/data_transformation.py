import sys, os, numpy as np, pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from healthapp.Cancer.constant.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from healthapp.Cancer.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from healthapp.exception.exception import HealthAppException
from healthapp.logging.logger import logging
from healthapp.Cancer.entity.config_entity import DataTransformationConfig
from healthapp.Cancer.utils.main_utils.utils import save_numpy_array_data, save_object

# The validated data now has exactly these columns.
FEATURE_COLS = [
    "concave points_mean", "area_mean", "radius_mean", "perimeter_mean", "concavity_mean"
]
# Expected target column.
TARGET_COLUMN = "diagnosis"

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise HealthAppException(e, sys)

    @staticmethod
    def read_data(file_path) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Loads validated data from the CSV file. Assumes the CSV contains exactly the columns:
            concave points_mean, area_mean, radius_mean, perimeter_mean, concavity_mean, diagnosis
        Returns:
            Xdata: DataFrame containing only the features.
            y_encoded: NumPy array with the encoded target (1 if 'M', 0 otherwise).
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"❌ Data file not found: {file_path}")
            df = pd.read_csv(file_path)
            # Clean column headers.
            df.columns = [col.strip() for col in df.columns]
            logging.info(f"Columns in validated data: {df.columns.tolist()}")
            
            # Verify required columns exist.
            required_cols = FEATURE_COLS + [TARGET_COLUMN]
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"❌ Missing columns in validated data: {missing_cols}")
            
            # Separate features and target.
            Xdata = df[FEATURE_COLS]
            ydata = df[TARGET_COLUMN]
            
            # Encode target: 'M' becomes 1, and any other value (typically 'B') becomes 0.
            y_encoded = np.asarray([1 if str(c).strip() == 'M' else 0 for c in ydata])
            return Xdata, y_encoded
        except Exception as e:
            raise HealthAppException(e, sys)

    def get_data_transformer_obj(self) -> Pipeline:
        """
        Initializes a preprocessing pipeline for numerical features:
            - KNNImputer: for filling missing values,
            - StandardScaler: to standardize features,
            - MinMaxScaler: to normalize features.
        """
        logging.info("Initializing transformation pipeline for numerical data...")
        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            scaler = StandardScaler()
            normalizer = MinMaxScaler()
            pipeline = Pipeline([
                ("imputer", imputer),
                ("scaler", scaler),
                ("normalizer", normalizer)
            ])
            return pipeline
        except Exception as e:
            raise HealthAppException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Executes the complete data transformation pipeline:
          1. Reads validated training and testing CSV files using paths from the data_validation artifact.
          2. Splits each into features and encoded target.
          3. Fits the transformation pipeline on training features.
          4. Transforms training and testing features.
          5. Concatenates the transformed features with their targets.
          6. Saves the transformed arrays and the preprocessor.
          7. Returns a DataTransformationArtifact with paths to the saved files.
        """
        logging.info("Starting cancer data transformation process...")
        try:
            # Load validated training and testing data.
            X_train_df, y_train = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            X_test_df, y_test = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            logging.info(f"Training data shape: {X_train_df.shape}, Testing data shape: {X_test_df.shape}")
            if X_train_df.empty or X_test_df.empty:
                raise ValueError("❌ Training or testing data is empty after validation.")
            
            # Fit the transformation pipeline on training features.
            transformer = self.get_data_transformer_obj()
            transformer_obj = transformer.fit(X_train_df)
            
            # Transform training and testing features.
            transformed_train = transformer_obj.transform(X_train_df)
            transformed_test = transformer_obj.transform(X_test_df)
            
            # Concatenate transformed features with target arrays.
            train_array = np.c_[transformed_train, y_train]
            test_array = np.c_[transformed_test, y_test]
            
            # Save transformed arrays and the preprocessor object.
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_array)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_array)
            save_object(self.data_transformation_config.transformed_object_file_path, transformer_obj)
            save_object("healthapp/Cancer/final_models/cancer_preprocessor.pkl", transformer_obj)
            logging.info("✅ Cancer preprocessor saved successfully!")
            
            artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            return artifact
        except Exception as e:
            raise HealthAppException(e, sys)