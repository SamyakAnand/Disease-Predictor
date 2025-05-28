from healthapp.exception.exception import HealthAppException
import os,sys
from healthapp.logging.logger import logging
import numpy as np
from healthapp.Diabetes.constant.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME

class NetworkModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor=preprocessor
            self.model=model
        except Exception as e:
            raise HealthAppException(e,sys)
    def predict(self,x):
        try:
            X_transform=self.preprocessor.transform(x)
            y_hat=self.model.predict(X_transform)
            return y_hat
        except Exception as e:
            raise HealthAppException(e,sys)