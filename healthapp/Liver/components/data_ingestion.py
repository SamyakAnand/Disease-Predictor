import sys, os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import pymongo

from healthapp.exception.exception import HealthAppException
from healthapp.app_logging.logger import logging
from healthapp.Liver.entity.config_entity import DataIngestionConfig
from healthapp.Liver.entity.artifact_entity import DataIngestionArtifact
from healthapp.Liver.utils.main_utils.utils import save_numpy_array_data, save_object

# Load environment variables (MongoDB credentials)
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

# Define expected feature columns and target column name
FEATURE_COLS = [
    'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 
    'Alamine_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio'
]
TARGET_COLUMN = "Dataset"

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise HealthAppException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Fetches data from MongoDB, cleans column names, and selects required columns.
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            # Connect to MongoDB
            mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = mongo_client[database_name][collection_name]

            # Load data
            df = pd.DataFrame(list(collection.find()))
            logging.info(f"Fetched {df.shape[0]} records from MongoDB.")

            # Clean column names
            df.columns = [col.strip() for col in df.columns]

            # Drop unnecessary MongoDB '_id' field if present
            if '_id' in df.columns:
                df.drop(columns=['_id'], inplace=True)

            # Handle missing values and alternative column names
            required_cols = FEATURE_COLS + [TARGET_COLUMN]
            missing_cols = set(required_cols) - set(df.columns)
            
            # Try fixing missing column names if they exist under alternative formats
            for col in missing_cols:
                alt_col = col.replace(" ", "_")  # Convert space-based names to underscores
                if alt_col in df.columns:
                    logging.info(f"Renaming column '{alt_col}' to '{col}'")
                    df.rename(columns={alt_col: col}, inplace=True)
            
            # Verify if required columns exist now
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"❌ Missing columns: {missing_cols}")

            # Select only required columns
            df = df[required_cols]

            # Convert Gender column to numeric if present
            if "Gender" in df.columns:
                df["Gender"] = df["Gender"].apply(lambda x: 1 if x.strip().lower() == "male" else 0)
                logging.info("Converted Gender column to numeric.")

            # Fill missing values with a constant (0.94 as placeholder)
            df.fillna(0.94, inplace=True)
            return df

        except Exception as e:
            raise HealthAppException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Saves the processed dataframe into a feature store.
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logging.info(f"Data saved into feature store at {feature_store_file_path}")
            return dataframe
        except Exception as e:
            raise HealthAppException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        """
        Splits the data into training and testing sets, then saves them.
        """
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=123
            )
            logging.info(f"Performed train-test split: Train ({train_set.shape}), Test ({test_set.shape})")

            # Save datasets
            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info("Exported train and test datasets successfully.")
        except Exception as e:
            raise HealthAppException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Executes the full data ingestion pipeline: fetching from MongoDB, saving to feature store, splitting.
        """
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            return data_ingestion_artifact
        except Exception as e:
            raise HealthAppException(e, sys)

if __name__ == "__main__":
    # Example usage (if needed for debugging):
    pass