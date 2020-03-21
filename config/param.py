import time
from config.load import TIME_DIR, IS_TRAIN
from lib.metrics import MultiLabelClassification as MLC

RANDOM_STATE = 42

measure_dict = {
    'accuracy': MLC.accuracy,
    'hamming_loss': MLC.hamming_loss,
    'f1': MLC.f1,
    'precision': MLC.precision,
    'recall': MLC.recall,
}

NEW_TIME_DIR = time.strftime('%Y_%m_%d_%H_%M_%S')
