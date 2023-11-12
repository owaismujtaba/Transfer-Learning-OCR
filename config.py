import os

CUR_DIR = os.getcwd()
print('Current Directory', CUR_DIR)

TRAIN_DATA_PATH = os.path.join(CUR_DIR,'data\csvTrainImages 13440x1024.csv' )
TEST_DATA_PATH = os.path.join(CUR_DIR, 'data\csvTestImages 3360x1024.csv')
TRAIN_DATA_LABELS = os.path.join(CUR_DIR, 'data\csvTrainLabel 13440x1.csv')
TEST_DATA_LABELS = os.path.join(CUR_DIR,'data\csvTestLabel 3360x1.csv')

TRAIN_VAL_SPLIT = 0.2
TRAIN = True
TEST = False