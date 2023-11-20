import os
import sys
import config
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_transformation import DataTransformationConfig
from src.utils import save_object, convert_tensor_to_dataset_loader
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import tensorflow as tf

import pdb

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(config.CUR_DIR, 'artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self, 
                    model, 
                    train_X, train_y, 
                    val_X, val_y, 
                    test_X, test_y,
                    optimizer, 
                 ):
        self.model_trainer_config = ModelTrainerConfig()
        self.model = model
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y
        self.test_X = test_X
        self.test_y = test_y
        self.optimizer = optimizer
    
        
    def train(self, name):
        try:
            logging.info('Starting Model vgg16 Training')
            early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
           
            #train_loader = convert_tensor_to_dataset_loader(self.train_X, self.train_y)
            #val_loader = convert_tensor_to_dataset_loader(self.val_X, self.val_y)
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            history = self.model.fit(
                self.train_X, self.train_y, 
                batch_size=config.BATCH_SIZE, 
                epochs=config.EPOCHS,
                validation_data=(self.val_X, self.val_y,),
                callbacks=[early_stopping]
            )
            logging.info(' Model Training Finised')
            logging.info('Saving the trained Model')
            
            trained_model_file_path = os.path.join(config.CUR_DIR, 'artifacts', name+'.h5')
            save_model(self.model, trained_model_file_path,)
            return history
        
        except Exception as e:
            raise CustomException(e, sys)
