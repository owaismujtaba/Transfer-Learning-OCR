import os
import sys
import config
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_transformation import DataTransformationConfig
from src.components.dataset_loader import DatasetLoader
from src.utils import save_object 

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import tensorflow as tf

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
            
            train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
            val_loss_metric = tf.keras.metrics.Mean(name='val_loss')

            # Training loop
            for epoch in range(config.EPOCHS):
                # Reset metrics at the start of each epoch
                train_loss_metric.reset_states()
                val_loss_metric.reset_states()
                
                train_loader  = DatasetLoader(self.train_X, self.train_y)
                
                for features, labels in train_loader:
                    
                    with tf.GradientTape() as tape:
                        predictions = self.model(features)
                        loss = self.loss_fn(labels, predictions)

                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                    # Update training loss metric
                    train_loss_metric.update_state(loss)
                
                val_loader = DatasetLoader(self.val_X, self.val_y)
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
        
    
        
        
class X:
   

    def train(self):
        # Define metrics for monitoring training
        train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        val_loss_metric = tf.keras.metrics.Mean(name='val_loss')

        # Training loop
        for epoch in range(self.epochs):
            # Reset metrics at the start of each epoch
            train_loss_metric.reset_states()
            val_loss_metric.reset_states()

            loader  = DatasetLoader(tr)
            # Training step
            for features, labels in self.train_dataset:
                with tf.GradientTape() as tape:
                    predictions = self.model(features)
                    loss = self.loss_fn(labels, predictions)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                # Update training loss metric
                train_loss_metric.update_state(loss)

            # Validation step
            for val_features, val_labels in self.val_dataset:
                val_predictions = self.model(val_features)
                val_loss = self.loss_fn(val_labels, val_predictions)

                # Update validation loss metric
                val_loss_metric.update_state(val_loss)

            # Print training and validation metrics for each epoch
            print(f'Epoch {epoch + 1}/{self.epochs}, '
                  f'Train Loss: {train_loss_metric.result():.4f}, '
                  f'Validation Loss: {val_loss_metric.result():.4f}')
