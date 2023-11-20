import pandas as pd
import tensorflow as tf
import config

class DatasetLoader(tf.data.Dataset):
    def __init__(self, X, y):
        self.batch_size = config.BATCH_SIZE
        self.features = X
        self.labels = y
        self.shuffle=True
        
        
        self.features = self.features.iloc[:, :-1].values
        self.labels = self.labels.iloc[:, -1].values

        # Create TensorFlow Dataset
        
        
        self.dataset = tf.data.Dataset.from_tensor_slices((self.features, self.labels))

        # Shuffle and batch the dataset
        if self.shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=len(self.features))
        self.dataset = self.dataset.batch(self.batch_size)

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataframe)