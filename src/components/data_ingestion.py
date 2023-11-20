import os
import sys
import pdb
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
import config
import pdb
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_x_data_path: str = os.path.join(config.CUR_DIR, r'src\artifacts\train_x.csv')
    train_y_data_path: str = os.path.join(config.CUR_DIR, r'src\artifacts\train_y.csv')
    val_x_data_path: str = os.path.join(config.CUR_DIR, r'src\artifacts\val_x.csv')
    val_y_data_path: str = os.path.join(config.CUR_DIR, r'src\artifacts\val_y.csv')
    test_x_data_path: str = os.path.join(config.CUR_DIR, r'src\artifacts\test_x.csv')
    test_y_data_path: str = os.path.join(config.CUR_DIR, r'src\artifacts\test_y.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion component')
        try:
            print('Data Ingestion Initiated')
            logging.info('Read the dataset as dataframe')
            train_X = pd.read_csv(config.TRAIN_DATA_PATH, header=None)
            train_y = pd.read_csv(config.TRAIN_DATA_LABELS, header=None)
            test_X = pd.read_csv(config.TEST_DATA_PATH, header=None)
            test_y = pd.read_csv(config.TEST_DATA_LABELS, header=None)
            print('Train Shape {} Test Shape {}'.format(train_X.shape, test_X.shape))
            
            logging.info('Split  train into train and valid')
            train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=config.TRAIN_VAL_SPLIT, random_state=42)
            train_X.to_csv(self.ingestion_config.train_x_data_path)
            train_y.to_csv(self.ingestion_config.train_y_data_path)
            test_X.to_csv(self.ingestion_config.test_x_data_path)
            test_y.to_csv(self.ingestion_config.test_y_data_path)
            val_X.to_csv(self.ingestion_config.val_x_data_path)
            val_y.to_csv(self.ingestion_config.val_y_data_path)
            print(train_X.shape, val_X.shape, test_X.shape)
            logging.info('Data Ingestion Completed')
            print('Data Ingestion finished')
            return(
                self.ingestion_config.train_x_data_path,
                self.ingestion_config.train_y_data_path,
                self.ingestion_config.val_x_data_path,
                self.ingestion_config.val_y_data_path,
                self.ingestion_config.test_x_data_path,
                self.ingestion_config.test_y_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        