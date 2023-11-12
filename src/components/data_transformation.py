import config
import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.components.data_ingestion import DataIngestionConfig
from src.exception import CustomException
from src.logger import logging

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
import pdb

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join(config.CUR_DIR, 'artifacts\preprocessor.pkl')
    

class ImagePreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = np.array(X)
        n_samples, _ = X.shape
        X = X.reshape(n_samples, 32, 32)
        return X
    

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_tranformer_obj(self):
        image_processor = Pipeline(
            steps=[
                ('shape_transform', ImagePreprocessor()),
                ('scaler', StandardScaler)
            ]
        )
        
        logging.info('Scaling values')
        logging.info('get data transformation completed')
        
        return image_processor

    def initiate_data_transformation(self):
        try:
            train_x = pd.read_csv(DataIngestionConfig.train_x_data_path)
            train_x = train_x.drop('Unnamed: 0', axis=1)
            val_x = pd.read_csv(DataIngestionConfig.val_x_data_path,)
            val_x = val_x.drop('Unnamed: 0', axis=1)
            test_x = pd.read_csv(DataIngestionConfig.test_x_data_path)
            test_x = test_x.drop('Unnamed: 0', axis=1)
            
            print(train_x.shape, val_x.shape, test_x.shape)
            pre_processor = self.get_data_tranformer_obj()
            logging.info('Appling preprocessing steps to train and test data')
            
            train_x = pre_processor.fit_transform(train_x)
            val_x = pre_processor.transform(val_x)
            test_x = pre_processor.transform(test_x)
            
        except Exception as e:
            raise CustomException(e, sys)
            
        
            
        