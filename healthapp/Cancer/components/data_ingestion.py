import sys, os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import pymongo

from healthapp.exception.exception import HealthAppException
from healthapp.logging.logger import logging
from healthapp.Cancer.entity.config_entity import DataIngestionConfig
from healthapp.Cancer.entity.artifact_entity import DataIngestionArtifact
from healthapp.Cancer.utils.main_utils.utils import save_numpy_array_data, save_object

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

# Define required columns exactly as desired.
# Note: "concave points_mean" must exactly match your expected column name.
FEATURE_COLS = ["concave points_mean", "area_mean", "radius_mean", "perimeter_mean", "concavity_mean"]
TARGET_COLUMN = "diagnosis"

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise HealthAppException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Reads data from MongoDB, cleans column names, and selects only the required 
        columns (FEATURE_COLS + [TARGET_COLUMN]).
        Instead of dropping columns by index, we now simply subset to the required columns.
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]
            
            # Load data from MongoDB.
            df = pd.DataFrame(list(collection.find()))
            print("MongoDB Columns:", df.columns.tolist())
            logging.info(f"MongoDB Columns: {df.columns.tolist()}")
            
            # Clean column names: remove extra spaces.
            df.columns = [col.strip() for col in df.columns]
            
            # Drop unwanted columns based on name rather than index.
            if '_id' in df.columns:
                df = df.drop(columns=['_id'])
            df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
            
            # Attempt to fix alternative naming,
            # e.g., if "concave points_mean" is stored as "concave_points_mean", rename it.
            required_cols = FEATURE_COLS + [TARGET_COLUMN]
            for col in required_cols:
                if col not in df.columns:
                    alt = col.replace(" ", "_")
                    if alt in df.columns:
                        logging.info(f"Renaming column '{alt}' to '{col}'")
                        df.rename(columns={alt: col}, inplace=True)
            
            missing_cols = set(required_cols) - set(df.columns.tolist())
            if missing_cols:
                raise ValueError(f"❌ Missing columns from CSV: {missing_cols}")
            
            # Instead of dropping columns by index, subset to the required columns.
            df = df[required_cols]
            logging.info(f"Dataframe shape after MongoDB fetch and cleaning: {df.shape}")
            
            df.replace({"na": np.nan}, inplace=True)
            return df
        except Exception as e:
            raise HealthAppException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Saves the processed dataframe to the feature store.
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
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
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train-test split on the dataframe.")
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            
            logging.info("Exported train and test datasets successfully.")
        except Exception as e:
            raise HealthAppException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Executes the full data ingestion pipeline: reads data from MongoDB,
        saves it to the feature store, splits it into train/test, and returns an artifact.
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
    # For example usage, instantiate your DataIngestionConfig and run the ingestion pipeline.
    pass