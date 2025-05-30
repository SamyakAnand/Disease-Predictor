from healthapp.Liver.components.data_ingestion import DataIngestion
from healthapp.Liver.components.data_validation import DataValidation
from healthapp.Liver.components.data_transformation import DataTransformation
from healthapp.Liver.components.model_trainer import ModelTrainer
from healthapp.exception.exception import HealthAppException
from healthapp.logging.logger import logging
import sys
from healthapp.Liver.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig
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
        
        data_transformation_config=DataTransformationConfig(trainingpipelineconfig)
        logging.info("data transformation started")
        data_transfromation=DataTransformation(data_validation_artifact,data_transformation_config)

        data_transfromation_artifact=data_transfromation.initiate_data_transformation()
        logging.info("data transformation completed")
        
        logging.info("Model Trainng started")
        model_trainer_config=ModelTrainerConfig(trainingpipelineconfig)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transfromation_artifact)
        model_trainer_artifact=model_trainer.initiate_model_trainer()    
    
    except Exception as e:
        raise HealthAppException(e,sys)