import os
import sys
import dill
import numpy as np
import pandas as pd
import config
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


from src.components.data_transformation import DataIngestionConfig
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def read_data_for_transformation():
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

            
            return train_X, train_y, val_X, val_y, test_X, test_y
    
    except Exception as e:
        raise CustomException(e, sys)