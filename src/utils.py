import os
import sys
import dill
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import config


import pdb
from src.components.data_ingestion import DataIngestionConfig
from src.exception import CustomException

def load_preprocessor_object(file_path):
    with open(file_path, 'rb') as file:
        loaded_obj = pickle.load(file)

    return loaded_obj

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def read_data_for_transformation(preprocessor):
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

            
            preprocessor = load_preprocessor_object(preprocessor)
            train_X  = preprocessor.transform(train_X)
            val_X = preprocessor.transform(val_X)
            test_X = preprocessor.transform(test_X)
            
            return train_X, train_y, val_X, val_y, test_X, test_y
    
    except Exception as e:
        raise CustomException(e, sys)
    
    
def convert_tensor_to_dataset_loader(X, y):
    features = X.iloc[:, :-1].values
    pdb.set_trace()
    labels = y.iloc[:, -1].values
            
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    dataset = dataset.shuffle(buffer_size=len(features))
    dataset = dataset.batch(config.BATCH_SIZE)

    return dataset