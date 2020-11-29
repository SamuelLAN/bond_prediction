import os
from config.load import MODEL_NAME

# ------------------------------------ data ------------------------------------------

# DATA_ROOT_DIR = r'D:\Data\share_mine_laptop\community_detection\data'
DATA_ROOT_DIR = r'/fs/clip-scratch/yusenlin/data'

# the directory that save the bond transaction data
TRACE_DIR = os.path.join(DATA_ROOT_DIR, 'pkl')

CACHE_DIR = os.path.join(DATA_ROOT_DIR, 'cache')

# the dict the map the bond_id/bond_index to bond_index/bond_id
DICT_BOND_ID_2_INDEX_JSON = os.path.join(DATA_ROOT_DIR, 'dict_bond_id_2_index.json')
DICT_BOND_INDEX_2_ID_JSON = os.path.join(DATA_ROOT_DIR, 'dict_bond_index_2_id.json')

D_DEALERS_TRACE_DIR = os.path.join(DATA_ROOT_DIR, 'd_dealers_trace')
D_BONDS_TRACE_DIR = os.path.join(DATA_ROOT_DIR, 'd_bonds_trace')

PROCESSED_DIR = os.path.join(DATA_ROOT_DIR, 'processed')

# ------------------------------- model -------------------------------------

TRAIN_MODEL_NAME = MODEL_NAME

__CUR_DIR = os.path.abspath(os.path.split(__file__)[0])
ROOT_DIR = os.path.split(__CUR_DIR)[0]

# directory for saving the runtime files
# __RUNTIME_DIR = os.path.join(ROOT_DIR, 'runtime')
__RUNTIME_DIR = r'/fs/clip-scratch/yusenlin/data/runtime/bond_prediction'

# directory for saving models
__PATH_MODEL_DIR = os.path.join(__RUNTIME_DIR, 'models')
PATH_MODEL_DIR = os.path.join(__PATH_MODEL_DIR, MODEL_NAME)

# directory for saving the tensorboard log files
__PATH_BOARD_DIR = os.path.join(__RUNTIME_DIR, 'tensorboard')
PATH_BOARD_DIR = os.path.join(__PATH_BOARD_DIR, MODEL_NAME)

PATH_TMP_DIR = os.path.join(__RUNTIME_DIR, 'tmp')

# the log file path, record all the models results and params
PATH_MODEL_LOG = os.path.join(__RUNTIME_DIR, 'model.log')
PATH_MODEL_LOG_DEALER = os.path.join(__RUNTIME_DIR, 'model_for_dealer_prediction.log')
PATH_CSV_LOG = os.path.join(__RUNTIME_DIR, 'experiments.csv')
PATH_CSV_LOG_DEALER = os.path.join(__RUNTIME_DIR, 'experiments_for_dealer_prediction.csv')


def mkdir_time(upper_path, _time):
    """ create directory with time (for save model) """
    dir_path = os.path.join(upper_path, _time)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return dir_path


def mk_if_not_exist(dir_path_list):
    """ create directory if not exist """
    for dir_path in dir_path_list:
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)


mk_if_not_exist([
    D_DEALERS_TRACE_DIR,
    D_BONDS_TRACE_DIR,
    __RUNTIME_DIR,
    __PATH_MODEL_DIR,
    PATH_MODEL_DIR,
    __PATH_BOARD_DIR,
    PATH_BOARD_DIR,
    PATH_TMP_DIR,
])
