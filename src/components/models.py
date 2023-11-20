import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten
import config
import pdb

class ModelVGG16:
    def __init__(self) -> None:
        
        self.vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(config.IMG_WIDTH,config.IMG_HEIGHT, 3))
        
        for layer in self.vgg16.layers:
            layer.trainable = False
        
        block5_output = self.vgg16.get_layer('block5_pool').output
        
        x = Flatten()(block5_output)
        x = Dense(2048, activation='relu')(x)   
        x = Dense(1024, activation='relu')(x)    
        x = Dense(215, activation='relu')(x)
        output_layer = Dense(config.OUTPUT_CLASSES, activation='softmax')(x)
        
        self.model = Model(inputs=self.vgg16.input, outputs=output_layer)
        
