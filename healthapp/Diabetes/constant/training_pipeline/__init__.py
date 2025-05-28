import os
import sys
import numpy as np
import pandas as pd

""" Creating necessary directories inside 'healthapp' automatically """
BASE_DIR = os.path.join("healthapp")  # Ensure healthapp is the base directory

# Define paths for required directories
ARTIFACT_DIR = os.path.join(BASE_DIR, "Artifacts")
DATA_SCHEMA_DIR = os.path.join(BASE_DIR, "data_schema")

# Ensure directories exist
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(DATA_SCHEMA_DIR, exist_ok=True)

""" Defining common constant variables for the training pipeline """
TARGET_COLUMN = 'Outcome'
PIPELINE_NAME: str = 'Health-App'
FILE_NAME: str = 'diabetes.csv'

TRAIN_FILE_NAME: str = 'train.csv'
TEST_FILE_NAME: str = 'test.csv'

SCHEMA_FILE_PATH = os.path.join(DATA_SCHEMA_DIR, "schema.yaml")

SAVED_MODEL_DIR = os.path.join("saved_models")
MODEL_FILE_NAME = "diabetes_model.pkl"

""" Data Ingestion related constants """
DATA_INGESTION_COLLECTION_NAME: str = "DiabetesData"
DATA_INGESTION_DATABASE_NAME: str = "HealthDB"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

""" Data Validation related constants """
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validation"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

""" Data Transformation related constants """
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

""" KNN imputer to replace NaN values """
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": 'uniform',
}

""" Model Trainer related constants """
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = 'trained_model'
MODEL_TRAINER_TRAINED_MODEL_NAME: str = 'diabetes_model.pkl'
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05