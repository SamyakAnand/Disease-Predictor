import os,sys
import json
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)
import certifi
ca=certifi.where()
import pandas as pd
import numpy as np
import pymongo
from pymongo import MongoClient
from healthapp.exception.exception import HealthAppException
from healthapp.logging.logger import logging

class HealthDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise HealthAppException(e,sys)
        
    def csv_to_json_convertor(self,file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise HealthAppException(e,sys)
        
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            
            self.database=self.mongo_client[self.database]
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return (len(self.records))
        except Exception as e:
            raise HealthAppException(e,sys)

if __name__ == "__main__":
    file_paths = {
        "data//cancer.csv": ("HealthDB", "CancerData"),
        "data//diabetes.csv": ("HealthDB", "DiabetesData"),
        "data//kidney.csv": ("HealthDB", "KidneyData"),
        "data//indian_liver_patient.csv": ("HealthDB", "LiverData"),
        "data//heart.csv": ("HealthDB", "HeartData"),
    }

    healthobj = HealthDataExtract()

    for file_path, (database, collection) in file_paths.items():
        records = healthobj.csv_to_json_convertor(file_path=file_path)
        no_of_records = healthobj.insert_data_mongodb(records, database, collection)
        print(f"Inserted {no_of_records} records into {collection} collection in {database} database.")