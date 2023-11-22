from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.trainer import ModelTrainer
from src.components.models import ModelVGG16
from src.utils import read_data_for_transformation
import config
import tensorflow as tf
from src.vis_utils import plot_accuracy, plot_loss
import pdb
import os

if __name__ == '__main__':
   if config.TRAIN:
        if config.DATA_INGESTION:
            ingest_data = DataIngestion()
            ingest_data.initiate_data_ingestion()
        else:
            print('Data Ingestion Already Done. Continuning to Transformation')
            
        if config.DATA_TRANSFORMATION:
            transform_data = DataTransformation()
            train_X, val_X, test_X, train_y, val_y, test_y = transform_data.initiate_data_transformation()
            pdb.set_trace()
        else:
            print('Data Transformation Already Done. Continuning to Transformation')
            
            train_X, train_y, val_X, val_y, test_X, test_y = read_data_for_transformation(preprocessor=DataTransformationConfig.preprocessor_obj_file)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)

        model = ModelVGG16()
        model.model.summary()
        
        model_train = ModelTrainer(model.model,
                                       train_X=train_X, train_y=train_y, 
                                       val_X=val_X, val_y=val_y,
                                       test_X=test_X, test_y=test_y,
                                       optimizer=optimizer
                                )
        
        dir_path = os.path.join(config.CUR_DIR, 'artifacts', 'Models')
        #dir_path = os.path.dirname(dir_path)
        os.makedirs(dir_path, exist_ok=True)
        
        model_name = 'vgg16v'
        version = 0
        
        while True:
            name = model_name + str(version) + '.h5'
            model_path = os.path.join(dir_path,name)
            if os.path.exists(model_path):
                version += 1
            else:
                break
        print(model_path)    
        fig_name = model_name.split('\\')[-1].split('.')[0] + str(version)
        #pdb.set_trace()
        print('fig;',fig_name)
        
        acc_plot_name = fig_name + '_accuracy.png'
        loss_plot_name = fig_name + '_loss.png'

        history = model_train.train(name=model_path)
        
        plot_accuracy(history=history, name=acc_plot_name)
        plot_loss(history=history, name=loss_plot_name)
        
        
       
