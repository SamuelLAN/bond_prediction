import time
from lib.metrics import MultiLabelClassification as MLC

RANDOM_STATE = 42
IS_TRAIN = True

measure_dict = {
    'accuracy': MLC.accuracy,
    'hamming_loss': MLC.hamming_loss,
    'f1': MLC.f1,
    'precision': MLC.precision,
    'recall': MLC.recall,
}

TIME_DIR = time.strftime('%Y_%m_%d_%H_%M_%S')
# TIME_DIR = '2020_01_08_18_16_13'
NEW_TIME_DIR = time.strftime('%Y_%m_%d_%H_%M_%S')
