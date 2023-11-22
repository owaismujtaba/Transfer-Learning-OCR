import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import tensorflow_hub as hub
import config
import pdb

class ModelVGG16:
    def __init__(self) -> None:
        
        self.vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(config.IMG_WIDTH,config.IMG_HEIGHT, 3))
        
        for layer in self.vgg16.layers:
            layer.trainable = False
        
        block3_output = self.vgg16.get_layer('block3_pool').output
        #trace()
        x = Conv2D(512, (3,3), activation='relu')(block3_output)
        x = Flatten()(x)
        x = Dense(2048, activation='relu')(x)   
        x = Dense(1024, activation='relu')(x)    
        x = Dense(215, activation='relu')(x)
        output_layer = Dense(config.OUTPUT_CLASSES, activation='softmax')(x)
        
        self.model = Model(inputs=self.vgg16.input, outputs=output_layer)
        
class InceptionModel:
    def __init__(self) -> None:
        
        self.model = InceptionV3(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
        for layer in self.model.layers:
            layer.trainable = False
        pdb.set_trace()
        for i, layer in enumerate(self.model.layers):
            print(f"Layer {i}: {layer.name} - Output Shape: {layer.output_shape}")
        pdb.set_trace()
        
        
InceptionModel()