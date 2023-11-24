import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

import pandas as pd

from src.logger import logging
from src.exception import CustomException


# To: Define the dataset paths, split the raw data into train and test set, and store accordingly


@dataclass
class DataIngestionConfig:
    trainData_path: str = os.path.join('artifacts', 'train.csv'),
    testData_path: str = os.path.join('artifacts', 'test.csv'),
    rawData_path: str = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.dataIngestionConfig = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info('Initiating Data Ingestion')
            logging.info('Reading Raw Data')
            raw_df = pd.read_csv("E:\\Projects\\Datasets\\StudentsPerformance.csv")
            os.makedirs(self.dataIngestionConfig.rawData_path, exist_ok=True)
            os.makedirs(self.dataIngestionConfig.trainData_path, exist_ok=True)
            os.makedirs(self.dataIngestionConfig.testData_path, exist_ok=True)

            raw_df.to_csv(self.dataIngestionConfig.rawData_path, index=False, header=True)

            train_set, test_set = train_test_split(raw_df, test_size=0.25, random_state=42 )
            logging.info('Train/Test data splitting completed.')

            train_set.to_csv(self.dataIngestionConfig.trainData_path, index=False, header=True)
            test_set.to_csv(self.dataIngestionConfig.testData_path, index=False, header=True)
            logging.info('Train/Test data saved.')

            return (self.dataIngestionConfig.trainData_path,
                    self.dataIngestionConfig.testData_path,
                    self.dataIngestionConfig.rawData_path)
        except Exception as e:
            raise CustomException(e, sys)


