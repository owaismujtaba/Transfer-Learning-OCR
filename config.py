import os

CUR_DIR = os.getcwd()
print('Current Directory', CUR_DIR)

TRAIN_DATA_PATH = os.path.join(CUR_DIR,'data\csvTrainImages 13440x1024.csv' )
TEST_DATA_PATH = os.path.join(CUR_DIR, 'data\csvTestImages 3360x1024.csv')
TRAIN_DATA_LABELS = os.path.join(CUR_DIR, 'data\csvTrainLabel 13440x1.csv')
TEST_DATA_LABELS = os.path.join(CUR_DIR,'data\csvTestLabel 3360x1.csv')

DATA_INGESTION = False
DATA_TRANSFORMATION = False

IMG_WIDTH = 32
IMG_HEIGHT = 32
OUTPUT_CLASSES = 28

TRAIN_VAL_SPLIT = 0.2
LEARNING_RATE = 0.0001
EPOCHS = 50
BATCH_SIZE = 32
TRAIN = True
TEST = False