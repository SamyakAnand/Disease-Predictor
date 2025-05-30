import sys, os
import mlflow.sklearn
import numpy as np
import pandas as pd
import mlflow
from healthapp.Liver.entity.artifact_entity import (
    DataTransformationArtifact,  
    DataValidationArtifact,
    ModelTrainerArtifact
)
from healthapp.exception.exception import HealthAppException
from healthapp.logging.logger import logging
from healthapp.Liver.entity.config_entity import DataTransformationConfig, ModelTrainerConfig
from healthapp.Liver.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models
)
from healthapp.Liver.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)

import dagshub
dagshub.init(repo_owner='SamyakAnand', repo_name='Health-App', mlflow=True)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise HealthAppException(e, sys)

    def track_mlflow(self, best_model, classification_metric):  
        """Logs model and metrics to MLflow."""
        with mlflow.start_run(run_name="Best Liver Model"):
            mlflow.log_params(best_model.get_params())
            mlflow.log_metric("F1 Score", classification_metric.f1_score)
            mlflow.log_metric("Precision", classification_metric.precision_score)
            mlflow.log_metric("Recall", classification_metric.recall_score)
            mlflow.sklearn.log_model(best_model, artifact_path="model")

    def train_model(self, X_train, y_train, X_test, y_test):
        """Trains multiple models and selects the best one."""
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, verbose=0),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, verbose=0),
            "Logistic Regression": LogisticRegression(verbose=0, max_iter=1000),
            "AdaBoost": AdaBoostClassifier(n_estimators=100),
            "SVC": SVC(kernel="rbf", C=1, gamma="scale")
        }

        params = {
            "Decision Tree": {"criterion": ["gini", "entropy"]},
            "Random Forest": {"n_estimators": [50, 100, 200]},
            "Gradient Boosting": {"learning_rate": [0.1, 0.05], "n_estimators": [50, 100]},
            "Logistic Regression": {"solver": ["lbfgs", "saga"]},
            "AdaBoost": {"n_estimators": [50, 100]},
            "SVC": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
        }

        # Evaluate models and pick the best one
        model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]

        logging.info(f"Best Model: {best_model_name}")

        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_train, y_train_pred)
        self.track_mlflow(best_model, classification_train_metric)

        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_test, y_test_pred)
        self.track_mlflow(best_model, classification_test_metric)

        preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
        save_object(self.model_trainer_config.trained_model_file_path, best_model)

        return ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """Loads data, splits it, and trains models."""
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # Validate paths before loading
            if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
                raise HealthAppException(f"File not found: {train_file_path} or {test_file_path}", sys)

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            return self.train_model(X_train, y_train, X_test, y_test)
        except Exception as e:
            raise HealthAppException(e, sys)