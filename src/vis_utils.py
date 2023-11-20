import matplotlib.pyplot as plt
import config
import os

def plot_accuracy(history, name):
    
    dir_path = os.path.join(config.CUR_DIR, 'artifacts', 'images')
    image_name = dir_path + name + '.png'
    
    dir_path = os.path.dirname(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.savefig(image_name, dpi=600)
    
def plot_loss(history, name):
    
    dir_path = os.path.join(config.CUR_DIR, 'artifacts', 'images')
    image_name = dir_path + name + '.png'
    
    dir_path = os.path.dirname(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.savefig(image_name, dpi=600)