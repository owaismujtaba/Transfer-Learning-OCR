import os
import sys
import config
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_transformation import DataTransformationConfig
from src.utils import save_object, convert_tensor_to_dataset_loader

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
    
        
    def train(self):
        try:
            logging.info('Starting Model Training')
            
            pdb.set_trace()
            train_loader = convert_tensor_to_dataset_loader(self.train_X, self.train_y)
            val_loader = convert_tensor_to_dataset_loader(self.val_X, self.val_y)
            
            
            train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
            val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
            pdb.set_trace()
            # Training loop
            for epoch in range(config.EPOCHS):
                # Reset metrics at the start of each epoch
                train_loss_metric.reset_states()
                val_loss_metric.reset_states()
                
                for features, labels in train_loader:
                    
                    with tf.GradientTape() as tape:
                        predictions = self.model(features)
                        loss = self.loss_fn(labels, predictions)

                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                    # Update training loss metric
                    train_loss_metric.update_state(loss)
                

                for features, labels in val_loader:
                    val_predictions = self.model(features)
                    val_loss = self.loss_fn(labels, val_predictions)

                    # Update validation loss metric
                    val_loss_metric.update_state(val_loss)
                
                print(f'Epoch {epoch + 1}/{self.epochs}, '
                  f'Train Loss: {train_loss_metric.result():.4f}, '
                  f'Validation Loss: {val_loss_metric.result():.4f}')

        except Exception as e:
            raise CustomException(e, sys)
