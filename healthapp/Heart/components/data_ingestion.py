import sys, os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import pymongo

from healthapp.exception.exception import HealthAppException
from healthapp.logging.logger import logging
from healthapp.Heart.entity.config_entity import DataIngestionConfig
from healthapp.Heart.entity.artifact_entity import DataIngestionArtifact
from healthapp.Heart.utils.main_utils.utils import save_numpy_array_data, save_object

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

# Define required feature columns to avoid passing unwanted ones
FEATURE_COLS = ['cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang']
TARGET_COLUMN = "target"

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise HealthAppException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """Reads data from MongoDB, filtering only required columns."""
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]

            # Fetch and select only required columns
            df = pd.DataFrame(list(collection.find()))[FEATURE_COLS + [TARGET_COLUMN]]

            logging.info(f"Dataframe shape after MongoDB fetch: {df.shape}")
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            df.replace({"na": np.nan}, inplace=True)
            return df
        except Exception as e:
            raise HealthAppException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Saves processed data to feature store."""
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise HealthAppException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        """Splits data into training and testing sets."""
        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train-test split on the dataframe.")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info("Exported train and test datasets successfully.")
        except Exception as e:
            raise HealthAppException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Runs the full data ingestion pipeline."""
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