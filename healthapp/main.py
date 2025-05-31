from healthapp.exception.exception import HealthAppException


import sys

def run_diabetes_pipeline():
    from healthapp.Diabetes.entity.config_entity import (
        DataIngestionConfig, TrainingPipelineConfig, 
        DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
    )
    from healthapp.Diabetes.components.data_ingestion import DataIngestion
    from healthapp.Diabetes.components.data_validation import DataValidation
    from healthapp.Diabetes.components.data_transformation import DataTransformation
    from healthapp.Diabetes.components.model_trainer import ModelTrainer

    try:
        
        # Set up overall training pipeline configuration
        training_pipeline_config = TrainingPipelineConfig()
        
        # Data Ingestion
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
        
        # Data Validation
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()
        print(data_validation_artifact)
        
        # Data Transformation
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        
        # Model Training
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        
    except Exception as e:
        raise HealthAppException(e, sys)


def run_kidney_pipeline():
    from healthapp.Kidney.entity.config_entity import (
        DataIngestionConfig, TrainingPipelineConfig,
        DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
    )
    from healthapp.Kidney.components.data_ingestion import DataIngestion
    from healthapp.Kidney.components.data_validation import DataValidation
    from healthapp.Kidney.components.data_transformation import DataTransformation
    from healthapp.Kidney.components.model_trainer import ModelTrainer

    trainingpipelineconfig = TrainingPipelineConfig()
    dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
    data_ingestion = DataIngestion(dataingestionconfig)
    dataingestionartifact = data_ingestion.initiate_data_ingestion()
    print(dataingestionartifact)
    
    data_validation_config = DataValidationConfig(trainingpipelineconfig)
    data_validation = DataValidation(dataingestionartifact, data_validation_config)
    data_validation_artifact = data_validation.initiate_data_validation()
    print(data_validation_artifact)
    
    data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
    data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
    data_transformation_artifact = data_transformation.initiate_data_transformation()
    
    model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
    model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
    model_trainer_artifact = model_trainer.initiate_model_trainer()
    
def run_liver_pipeline():
    from healthapp.Liver.entity.config_entity import (
        DataIngestionConfig, TrainingPipelineConfig,
        DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
    )
    from healthapp.Liver.components.data_ingestion import DataIngestion
    from healthapp.Liver.components.data_validation import DataValidation
    from healthapp.Liver.components.data_transformation import DataTransformation
    from healthapp.Liver.components.model_trainer import ModelTrainer

    trainingpipelineconfig = TrainingPipelineConfig()
    dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
    data_ingestion = DataIngestion(dataingestionconfig)
    dataingestionartifact = data_ingestion.initiate_data_ingestion()
    print(dataingestionartifact)
    
    data_validation_config = DataValidationConfig(trainingpipelineconfig)
    data_validation = DataValidation(dataingestionartifact, data_validation_config)
    data_validation_artifact = data_validation.initiate_data_validation()
    print(data_validation_artifact)
    
    data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
    data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
    data_transformation_artifact = data_transformation.initiate_data_transformation()
    
    model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
    model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
    model_trainer_artifact = model_trainer.initiate_model_trainer()
    
def run_cancer_pipeline():
    from healthapp.Cancer.entity.config_entity import (
        DataIngestionConfig, TrainingPipelineConfig,
        DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
    )
    from healthapp.Cancer.components.data_ingestion import DataIngestion
    from healthapp.Cancer.components.data_validation import DataValidation
    from healthapp.Cancer.components.data_transformation import DataTransformation
    from healthapp.Cancer.components.model_trainer import ModelTrainer

    trainingpipelineconfig = TrainingPipelineConfig()
    dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
    data_ingestion = DataIngestion(dataingestionconfig)
    dataingestionartifact = data_ingestion.initiate_data_ingestion()
    print(dataingestionartifact)
    
    data_validation_config = DataValidationConfig(trainingpipelineconfig)
    data_validation = DataValidation(dataingestionartifact, data_validation_config)
    data_validation_artifact = data_validation.initiate_data_validation()
    print(data_validation_artifact)
    
    data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
    data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
    data_transformation_artifact = data_transformation.initiate_data_transformation()
    
    model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
    model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
    model_trainer_artifact = model_trainer.initiate_model_trainer()
    
def run_heart_pipeline():
    from healthapp.Heart.entity.config_entity import (
        DataIngestionConfig, TrainingPipelineConfig,
        DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
    )
    from healthapp.Heart.components.data_ingestion import DataIngestion
    from healthapp.Heart.components.data_validation import DataValidation
    from healthapp.Heart.components.data_transformation import DataTransformation
    from healthapp.Heart.components.model_trainer import ModelTrainer

    trainingpipelineconfig = TrainingPipelineConfig()
    dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
    data_ingestion = DataIngestion(dataingestionconfig)
    dataingestionartifact = data_ingestion.initiate_data_ingestion()
    print(dataingestionartifact)
    
    data_validation_config = DataValidationConfig(trainingpipelineconfig)
    data_validation = DataValidation(dataingestionartifact, data_validation_config)
    data_validation_artifact = data_validation.initiate_data_validation()
    print(data_validation_artifact)
    
    data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
    data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
    data_transformation_artifact = data_transformation.initiate_data_transformation()
    
    model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
    model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
    model_trainer_artifact = model_trainer.initiate_model_trainer()

if __name__=='__main__':
    try:
        run_diabetes_pipeline()
        run_kidney_pipeline()
        run_liver_pipeline()
        run_cancer_pipeline()
        run_heart_pipeline()
    except Exception as e:
        raise HealthAppException(e, sys)