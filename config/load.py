import os
from config import path

TRAIN_VAL_RATIO = 0.995

# ------------------------ (interval input, bond prediction) for loading single dealer data --------------------------

DATA_SUBSET = path.PREDICTION_DATE_BY_VOLUME_DEALERS_BY_BOND_SELL_CLIENT
YEAR = '2015'
VOLUME_LEVEL = '100_1k'
DATA_INDEX = 'bond_AES3903624'
NO_BELOW = 5

# ------------------------ (temporal input, bond prediction) for loading single dealer data --------------------------

DATA_SUBSET_FOR_TEMPORAL_INPUT = path.PREDICTION_DATE_BY_VOLUME_BONDS_BY_DEALER_DIR
YEAR_FOR_TEMPORAL_INPUT = '2015'
VOLUME_LEVEL_FOR_TEMPORAL_INPUT = '10k_100k'
DATA_INDEX_FOR_TEMPORAL_INPUT = 'dealer_83'
NO_BELOW_FOR_TEMPORAL_INPUT = 10


# ------------------------ (interval input, dealer prediction) for loading single dealer data --------------------------

DATA_SUBSET_FOR_DEALER_PRED = path.PREDICTION_DATE_BY_VOLUME_DEALERS_BY_BOND_DIR
YEAR_FOR_DEALER_PRED = '2015'
VOLUME_LEVEL_FOR_DEALER_PRED = '10k_100k'
DATA_INDEX_FOR_DEALER_PRED = 'bond_AAPL4001809'
NO_BELOW_FOR_DEALER_PRED = 10

# ---------------------- for bond embeddings -----------------------

NO_BELOW_BONDS_FOR_EMB = 1000

DATA_DIR_LIST = [
    path.ORIGIN_VOLUME_1k_10k_DIR,
    path.ORIGIN_VOLUME_10k_100k_DIR,
    path.ORIGIN_VOLUME_100k_1m_DIR,
    path.ORIGIN_VOLUME_1m_10m_DIR,
    path.ORIGIN_VOLUME_10m_DIR,
]

# ---------------------- for log ---------------------------

# LOG = {
#     'subset': os.path.split(DATA_SUBSET)[1],
#     'year': YEAR,
#     'volume_level': VOLUME_LEVEL,
#     'data_index': DATA_INDEX,
#     'no_below': NO_BELOW,
#     'train/test_ratio': '0.8/0.2',
# }

LOG = {
    'subset': os.path.split(DATA_SUBSET_FOR_TEMPORAL_INPUT)[1],
    'year': YEAR_FOR_TEMPORAL_INPUT,
    'volume_level': VOLUME_LEVEL_FOR_TEMPORAL_INPUT,
    'data_index': DATA_INDEX_FOR_TEMPORAL_INPUT,
    'no_below': NO_BELOW_FOR_TEMPORAL_INPUT,
    'train/test_ratio': '0.8/0.2',
}
