import matplotlib.pyplot as plt
import config
import os
import pdb

def plot_accuracy(history, name):
    
    dir_path = os.path.join(config.CUR_DIR, 'artifacts', 'images')
    os.makedirs(dir_path, exist_ok=True)
    image_name = os.path.join(dir_path, name)
    
    pdb.set_trace()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.savefig(image_name, dpi=600)
    plt.clf()
    
def plot_loss(history, name):
    
    dir_path = os.path.join(config.CUR_DIR, 'artifacts', 'Images')
    os.makedirs(dir_path, exist_ok=True)
    image_name = os.path.join(dir_path, name)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.savefig(image_name, dpi=600)
    plt.clf()