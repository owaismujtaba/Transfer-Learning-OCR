import config
import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.components.data_ingestion import DataIngestionConfig
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
import pdb

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join(config.CUR_DIR, 'artifacts', 'preprocessor.pkl')
    

class ImagePreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = np.array(X)
        n_samples, _ = X.shape
        X = X.reshape(n_samples, 32, 32, 1)
        X = np.repeat(X, 3, axis=3)
        return X
    

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_tranformer_obj(self):
        image_processor = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('shape_transform', ImagePreprocessor())
            ]
        )
        
        logging.info('Scaling values')
        logging.info('get data transformation completed')
        
        return image_processor

    def initiate_data_transformation(self):
        try:
            print('Data transformation Initiated')
            
            train_X = pd.read_csv(DataIngestionConfig.train_x_data_path)
            train_y = pd.read_csv(DataIngestionConfig.train_y_data_path)
            train_X = train_X.drop('Unnamed: 0', axis=1)
            train_y = train_y.drop('Unnamed: 0', axis=1)
            
            val_X = pd.read_csv(DataIngestionConfig.val_x_data_path)
            val_y = pd.read_csv(DataIngestionConfig.val_y_data_path)
            val_y = val_y.drop('Unnamed: 0', axis=1)
            val_X = val_X.drop('Unnamed: 0', axis=1)
            
            test_X = pd.read_csv(DataIngestionConfig.test_x_data_path)
            test_y = pd.read_csv(DataIngestionConfig.test_y_data_path)
            test_X = test_X.drop('Unnamed: 0', axis=1)
            test_y = test_y.drop('Unnamed: 0', axis=1)
            
            print(train_X.shape, val_X.shape, test_X.shape)
            pre_processor = self.get_data_tranformer_obj()
            logging.info('Appling preprocessing steps to train and test data')
            
            train_X = pre_processor.fit_transform(train_X)
            val_X = pre_processor.transform(val_X)
            test_X = pre_processor.transform(test_X)
            
            #train_y, val_y, test_y  = encode_labels_logits(train_y, val_y, test_y)
            
            print(train_X.shape, val_X.shape, test_X.shape)
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file, obj=pre_processor)
            print('Saved preprocessor object')
            logging.info('Preprocessing of the filesd done and preprocessor saveds')
            print('Data Transformation Completed')
            
            
            return train_X, val_X, test_X, train_y, val_y, test_y
            
        except Exception as e:
            raise CustomException(e, sys)
            
        
            
        