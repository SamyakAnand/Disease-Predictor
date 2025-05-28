from healthapp.Diabetes.components.data_ingestion import DataIngestion
from healthapp.Diabetes.components.data_validation import DataValidation
from healthapp.Diabetes.components.data_transformation import DataTransformation
from healthapp.Diabetes.components.model_trainer import ModelTrainer
from healthapp.exception.exception import HealthAppException
from healthapp.logging.logger import logging
import sys
from healthapp.Diabetes.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig
if __name__=='__main__':
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        print(dataingestionartifact)
        data_validation_config=DataValidationConfig(trainingpipelineconfig)
        data_validation=DataValidation(dataingestionartifact,data_validation_config)
        logging.info("Initiate the data validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("Completed the data validation")
        print(data_validation_artifact)
        
        
    except Exception as e:
        raise HealthAppException(e,sys)