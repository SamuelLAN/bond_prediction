import time
from lib.metrics import MultiLabelClassification as MLC

RANDOM_STATE = 42
IS_TRAIN = False

measure_dict = {
    'accuracy': MLC.accuracy,
    'hamming_loss': MLC.hamming_loss,
    'f1': MLC.f1,
    'precision': MLC.precision,
    'recall': MLC.recall,
}

# TIME_DIR = time.strftime('%Y_%m_%d_%H_%M_%S')
TIME_DIR = '2020_03_14_00_27_57'
NEW_TIME_DIR = time.strftime('%Y_%m_%d_%H_%M_%S')
