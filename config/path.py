import os

# ------------------------------------ data ------------------------------------------

DATA_ROOT_DIR = r'D:\Data\share_mine_laptop\community_detection\data'

# the directory that save the bond transaction data
TRACE_DIR = os.path.join(DATA_ROOT_DIR, 'pkl')

TOPIC_BONDS_LIST_JSON = os.path.join(DATA_ROOT_DIR, 'topics_bond_weight_list.json')
TOPIC_BONDS_JSON = os.path.join(DATA_ROOT_DIR, 'topics_bonds_for_prediction.json')

PREDICTION_DIR = os.path.join(DATA_ROOT_DIR, 'prediction')
PREDICTION_BONDS_BY_DEALER_DIR = os.path.join(PREDICTION_DIR, 'bonds_by_dealer')
PREDICTION_BONDS_BY_DEALER_BUY_CLIENT = os.path.join(PREDICTION_DIR, 'bonds_by_dealer_buy_client')
PREDICTION_BONDS_BY_DEALER_SELL_CLIENT = os.path.join(PREDICTION_DIR, 'bonds_by_dealer_sell_client')
PREDICTION_BONDS_BY_DEALER_BUY_DEALER = os.path.join(PREDICTION_DIR, 'bonds_by_dealer_buy_dealer')
PREDICTION_BONDS_BY_DEALER_SELL_DEALER = os.path.join(PREDICTION_DIR, 'bonds_by_dealer_sell_dealer')
PREDICTION_BONDS_BY_DEALER_CLIENTS = os.path.join(PREDICTION_DIR, 'bonds_by_dealer_clients')
PREDICTION_BONDS_BY_DEALER_DEALERS = os.path.join(PREDICTION_DIR, 'bonds_by_dealer_dealers')

PREDICTION_DEALERS_BY_BOND_DIR = os.path.join(PREDICTION_DIR, 'dealers_by_bond')
PREDICTION_DEALERS_BY_BOND_BUY_CLIENT = os.path.join(PREDICTION_DIR, 'dealers_by_bond_buy_client')
PREDICTION_DEALERS_BY_BOND_SELL_CLIENT = os.path.join(PREDICTION_DIR, 'dealers_by_bond_sell_client')
PREDICTION_DEALERS_BY_BOND_CLIENTS = os.path.join(PREDICTION_DIR, 'dealers_by_bond_clients')
PREDICTION_DEALERS_BY_BOND_DEALERS = os.path.join(PREDICTION_DIR, 'dealers_by_bond_dealers')

PREDICTION_DATE_DIR = os.path.join(DATA_ROOT_DIR, 'prediction_date')
PREDICTION_DATE_BONDS_BY_DEALER_DIR = os.path.join(PREDICTION_DATE_DIR, 'bonds_by_dealer')
PREDICTION_DATE_BONDS_BY_DEALER_BUY_CLIENT = os.path.join(PREDICTION_DATE_DIR, 'bonds_by_dealer_buy_client')
PREDICTION_DATE_BONDS_BY_DEALER_SELL_CLIENT = os.path.join(PREDICTION_DATE_DIR, 'bonds_by_dealer_sell_client')
PREDICTION_DATE_BONDS_BY_DEALER_DEALER = os.path.join(PREDICTION_DATE_DIR, 'bonds_by_dealer_dealer')

PREDICTION_DATE_DEALERS_BY_BOND_DIR = os.path.join(PREDICTION_DATE_DIR, 'dealers_by_bond')
PREDICTION_DATE_DEALERS_BY_BOND_BUY_CLIENT = os.path.join(PREDICTION_DATE_DIR, 'dealers_by_bond_buy_client')
PREDICTION_DATE_DEALERS_BY_BOND_SELL_CLIENT = os.path.join(PREDICTION_DATE_DIR, 'dealers_by_bond_sell_client')
PREDICTION_DATE_DEALERS_BY_BOND_DEALER = os.path.join(PREDICTION_DATE_DIR, 'dealers_by_bond_dealer')

PREDICTION_DATE_BY_VOLUME_DIR = os.path.join(DATA_ROOT_DIR, 'prediction_date_by_level')
PREDICTION_DATE_BY_VOLUME_BONDS_BY_DEALER_DIR = os.path.join(PREDICTION_DATE_BY_VOLUME_DIR, 'bonds_by_dealer')
PREDICTION_DATE_BY_VOLUME_BONDS_BY_DEALER_BUY_CLIENT = os.path.join(PREDICTION_DATE_BY_VOLUME_DIR, 'bonds_by_dealer_buy_client')
PREDICTION_DATE_BY_VOLUME_BONDS_BY_DEALER_SELL_CLIENT = os.path.join(PREDICTION_DATE_BY_VOLUME_DIR, 'bonds_by_dealer_sell_client')
PREDICTION_DATE_BY_VOLUME_BONDS_BY_DEALER_DEALER = os.path.join(PREDICTION_DATE_BY_VOLUME_DIR, 'bonds_by_dealer_dealer')

PREDICTION_DATE_BY_VOLUME_DEALERS_BY_BOND_DIR = os.path.join(PREDICTION_DATE_BY_VOLUME_DIR, 'dealers_by_bond')
PREDICTION_DATE_BY_VOLUME_DEALERS_BY_BOND_BUY_CLIENT = os.path.join(PREDICTION_DATE_BY_VOLUME_DIR, 'dealers_by_bond_buy_client')
PREDICTION_DATE_BY_VOLUME_DEALERS_BY_BOND_SELL_CLIENT = os.path.join(PREDICTION_DATE_BY_VOLUME_DIR, 'dealers_by_bond_sell_client')
PREDICTION_DATE_BY_VOLUME_DEALERS_BY_BOND_DEALER = os.path.join(PREDICTION_DATE_BY_VOLUME_DIR, 'dealers_by_bond_dealer')

# the directory for saving the bonds which are grouped by dealer index
BONDS_BY_DEALER_DIR = os.path.join(DATA_ROOT_DIR, 'bonds_by_dealer')
BONDS_BY_DEALER_DATE_DOC_DIR = os.path.join(DATA_ROOT_DIR, 'bonds_by_dealer_date_doc')

# the dict the map the bond_id/bond_index to bond_index/bond_id
DICT_BOND_ID_2_INDEX_JSON = os.path.join(DATA_ROOT_DIR, 'dict_bond_id_2_index.json')
DICT_BOND_INDEX_2_ID_JSON = os.path.join(DATA_ROOT_DIR, 'dict_bond_index_2_id.json')

# ------------------------------- model -------------------------------------

MODEL_NAME = 'lstm_new'
TRAIN_MODEL_NAME = MODEL_NAME

__CUR_DIR = os.path.abspath(os.path.split(__file__)[0])
ROOT_DIR = os.path.split(__CUR_DIR)[0]

# directory for saving the runtime files
__RUNTIME_DIR = os.path.join(ROOT_DIR, 'runtime')

# directory for saving models
__PATH_MODEL_DIR = os.path.join(__RUNTIME_DIR, 'models')
PATH_MODEL_DIR = os.path.join(__PATH_MODEL_DIR, MODEL_NAME)

# directory for saving the tensorboard log files
__PATH_BOARD_DIR = os.path.join(__RUNTIME_DIR, 'tensorboard')
PATH_BOARD_DIR = os.path.join(__PATH_BOARD_DIR, MODEL_NAME)

PATH_TMP_DIR = os.path.join(__RUNTIME_DIR, 'tmp')

# the log file path, record all the models results and params
PATH_MODEL_LOG = os.path.join(__RUNTIME_DIR, 'model.log')
PATH_CSV_LOG = os.path.join(__RUNTIME_DIR, 'experiments.csv')


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
    __RUNTIME_DIR,
    __PATH_MODEL_DIR,
    PATH_MODEL_DIR,
    __PATH_BOARD_DIR,
    PATH_BOARD_DIR,
    PATH_TMP_DIR,
    PREDICTION_DIR,
    PREDICTION_BONDS_BY_DEALER_DIR,
    PREDICTION_BONDS_BY_DEALER_BUY_CLIENT,
    PREDICTION_BONDS_BY_DEALER_SELL_CLIENT,
    PREDICTION_BONDS_BY_DEALER_BUY_DEALER,
    PREDICTION_BONDS_BY_DEALER_SELL_DEALER,
    PREDICTION_BONDS_BY_DEALER_CLIENTS,
    PREDICTION_BONDS_BY_DEALER_DEALERS,
    PREDICTION_DEALERS_BY_BOND_DIR,
    PREDICTION_DEALERS_BY_BOND_BUY_CLIENT,
    PREDICTION_DEALERS_BY_BOND_SELL_CLIENT,
    PREDICTION_DEALERS_BY_BOND_CLIENTS,
    PREDICTION_DEALERS_BY_BOND_DEALERS,
    PREDICTION_DATE_DIR,
    PREDICTION_DATE_BONDS_BY_DEALER_DIR,
    PREDICTION_DATE_BONDS_BY_DEALER_BUY_CLIENT,
    PREDICTION_DATE_BONDS_BY_DEALER_SELL_CLIENT,
    PREDICTION_DATE_BONDS_BY_DEALER_DEALER,
    PREDICTION_DATE_DEALERS_BY_BOND_DIR,
    PREDICTION_DATE_DEALERS_BY_BOND_BUY_CLIENT,
    PREDICTION_DATE_DEALERS_BY_BOND_SELL_CLIENT,
    PREDICTION_DATE_DEALERS_BY_BOND_DEALER,
    PREDICTION_DATE_BY_VOLUME_DIR,
    PREDICTION_DATE_BY_VOLUME_BONDS_BY_DEALER_DIR,
    PREDICTION_DATE_BY_VOLUME_BONDS_BY_DEALER_BUY_CLIENT,
    PREDICTION_DATE_BY_VOLUME_BONDS_BY_DEALER_SELL_CLIENT,
    PREDICTION_DATE_BY_VOLUME_BONDS_BY_DEALER_DEALER,
    PREDICTION_DATE_BY_VOLUME_DEALERS_BY_BOND_DIR,
    PREDICTION_DATE_BY_VOLUME_DEALERS_BY_BOND_BUY_CLIENT,
    PREDICTION_DATE_BY_VOLUME_DEALERS_BY_BOND_SELL_CLIENT,
    PREDICTION_DATE_BY_VOLUME_DEALERS_BY_BOND_DEALER,
])
