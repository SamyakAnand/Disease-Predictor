import sys, os, numpy as np, pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from healthapp.Kidney.constant.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from healthapp.Kidney.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from healthapp.exception.exception import HealthAppException
from healthapp.logging.logger import logging
from healthapp.Kidney.entity.config_entity import DataTransformationConfig
from healthapp.Kidney.utils.main_utils.utils import save_numpy_array_data, save_object

# For Kidney dataset process, our transformation features are defined as follows:
FEATURE_COLS = ['bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc']

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise HealthAppException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Loads data from the CSV file and cleans it by applying replacement operations for categorical fields.
        The cleaning steps include:
          - For columns ['htn','dm','cad','pe','ane']: replace {'yes':1, 'no':0} if available.
          - For columns ['rbc','pc']: replace {'abnormal':1, 'normal':0} if available.
          - For columns ['pcc','ba']: replace {'present':1,'notpresent':0} if available.
          - For column 'appet': replace {'good':1, 'poor':0, 'no':np.nan} if available.
          - For 'classification': replace {'ckd':1.0, 'ckd\t':1.0, 'notckd':0.0, 'no':0.0} and then rename to 'class'.
          - Additional corrections for columns 'pe', 'appet', 'cad', and 'dm'.
          - Drops the 'id' column (if found) and then drops rows with any missing values.
          
        Finally, it returns only the required columns: FEATURE_COLS + [TARGET_COLUMN].
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"❌ Data file not found: {file_path}")
            df = pd.read_csv(file_path)
            df.columns = [col.strip() for col in df.columns]

            # Replace values in columns if they exist.
            cols_to_replace = ['htn', 'dm', 'cad', 'pe', 'ane']
            available = [col for col in cols_to_replace if col in df.columns]
            if available:
                df[available] = df[available].replace(to_replace={'yes': 1, 'no': 0})
            else:
                logging.warning("Columns for replacement not found: " + str(set(cols_to_replace) - set(df.columns)))
            
            # For 'rbc' and 'pc'
            for col in ['rbc', 'pc']:
                if col in df.columns:
                    df[col] = df[col].replace({'abnormal': 1, 'normal': 0})
            
            # For 'pcc' and 'ba'
            for col in ['pcc', 'ba']:
                if col in df.columns:
                    df[col] = df[col].replace({'present': 1, 'notpresent': 0})
            
            # For 'appet'
            if 'appet' in df.columns:
                df['appet'] = df['appet'].replace({'good': 1, 'poor': 0, 'no': np.nan})
            
            # For 'classification', then rename to 'class'
            if 'classification' in df.columns:
                df['classification'] = df['classification'].replace({'ckd': 1.0, 'ckd\t': 1.0, 'notckd': 0.0, 'no': 0.0})
                df.rename(columns={'classification': TARGET_COLUMN}, inplace=True)
            # Additional corrections:
            if 'pe' in df.columns:
                df['pe'] = df['pe'].replace('good', 0)
            if 'appet' in df.columns:
                df['appet'] = df['appet'].replace('no', 0)
            if 'cad' in df.columns:
                df['cad'] = df['cad'].replace('\tno', 0)
            if 'dm' in df.columns:
                df['dm'] = df['dm'].replace({'\tno': 0, '\tyes': 1, ' yes': 1, '': np.nan})
            
            # Drop the 'id' column if present.
            if 'id' in df.columns:
                df.drop('id', axis=1, inplace=True)
            
            # Drop rows with missing values.
            df = df.dropna(axis=0)

            # Verify that the final DataFrame contains only the required columns.
            required_cols = FEATURE_COLS + [TARGET_COLUMN]
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"❌ Missing columns from CSV: {missing_cols}")
            
            # Return only the required columns.
            return df[required_cols]
        except Exception as e:
            raise HealthAppException(e, sys)

    def get_data_transformer_obj(self) -> Pipeline:
        """
        Initializes a preprocessing pipeline that consists of:
          - KNNImputer (to handle any missing values),
          - StandardScaler (to standardize features),
          - MinMaxScaler (to scale the standardized data).
        """
        logging.info("Initializing preprocessing pipeline for numerical data transformation.")
        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            scaler = StandardScaler()
            normalizer = MinMaxScaler()
            processor = Pipeline([
                ("imputer", imputer),
                ("scaler", scaler),
                ("normalizer", normalizer)
            ])
            return processor
        except Exception as e:
            raise HealthAppException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Executes the full data transformation pipeline:
         - Loads the training and testing data after cleaning.
         - Separates features (as defined in FEATURE_COLS) and the target (TARGET_COLUMN, i.e. "class").
         - Fits and applies the numerical transformation pipeline to the features.
         - Concatenates the transformed features with the target.
         - Saves the transformed train/test arrays and the preprocessor object.
        """
        logging.info("Starting numerical data transformation process...")
        try:
            # Load the validated (and cleaned) training and testing data.
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)
            
            logging.info(f"Train Data Shape (after filtering & cleaning): {train_df.shape}, "
                         f"Test Data Shape: {test_df.shape}")
            
            if train_df.empty or test_df.empty:
                raise ValueError("❌ Train or Test dataset is empty. Check data ingestion.")
            
            # Separate input features and target.
            input_feature_train_df = train_df[FEATURE_COLS]
            target_feature_train_df = train_df[TARGET_COLUMN]
            
            input_feature_test_df = test_df[FEATURE_COLS]
            target_feature_test_df = test_df[TARGET_COLUMN]
            
            # Fit the transformation pipeline on the training features.
            preprocessor = self.get_data_transformer_obj()
            preprocessor_obj = preprocessor.fit(input_feature_train_df)
            
            transformed_train_features = preprocessor_obj.transform(input_feature_train_df)
            transformed_test_features = preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[transformed_train_features, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_test_features, np.array(target_feature_test_df)]
            
            # Save the transformed arrays.
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            
            # Save the preprocessor object to the configured location and a dedicated Kidney folder.
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_obj)
            save_object("healthapp/Kidney/final_models/kidney_preprocessor.pkl", preprocessor_obj)
            
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact
        except Exception as e:
            raise HealthAppException(e, sys)