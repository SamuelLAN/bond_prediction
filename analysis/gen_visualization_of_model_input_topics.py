import os
import numpy as np
from matplotlib import pyplot as plt
from config import path, date
from lib import utils
from analysis import gen_inputs_2 as gen_inputs

data = []


name_map = {
    'buy_client': 'BfC',
    'sell_client': 'StC',
    'buy_dealer': 'BfD',
    'sell_dealer': 'StD',
    'clients': 'BfC+StC',
    'dealers': 'BfD+StD',
    '': 'all',
}

dir_name = 'bonds_by_dealer'
dir_path = os.path.join(path.PREDICTION_DATE_DIR, dir_name, '2015')

name = dir_name.replace('bonds_by_dealer_', '').replace('bonds_by_dealer', '')
name = name_map[name]

save_dir_path = os.path.join(path.ROOT_DIR, 'runtime', 'visualization_of_model_inputs_topics')
if not os.path.isdir(save_dir_path):
    os.mkdir(save_dir_path)

transaction_count_levels = ['100_1k', '1k_10k', '10k_100k', '100k_1m']
for _dir_name in transaction_count_levels:
    new_dir_path = os.path.join(save_dir_path, _dir_name)
    if not os.path.isdir(new_dir_path):
        os.mkdir(new_dir_path)

path_dict_bond_id_offering_date = os.path.join(path.DATA_ROOT_DIR, 'dict_bond_id_offering_date.json')
dict_bond_id_offering_date = utils.load_json(path_dict_bond_id_offering_date)
remove_bond_list = []

bound_date = '2014-12-01'
bound_timestamp = utils.date_2_timestamp(bound_date)
for bond_id, offering_date in dict_bond_id_offering_date.items():
    if offering_date[:2] == '19':
        continue
    elif offering_date[:1] != '2':
        remove_bond_list.append(bond_id)
        continue

    offering_date = offering_date.replace('-00', '-01')

    offering_timestamp = utils.date_2_timestamp(offering_date)
    if offering_timestamp >= bound_timestamp:
        remove_bond_list.append(bond_id)

tmp_dir_path = os.path.join(path.PREDICTION_DATE_BY_VOLUME_DIR, dir_name, '2015')
d_dealers = {
    '100_1k': os.listdir(os.path.join(tmp_dir_path, '100_1k')),
    '1k_10k': os.listdir(os.path.join(tmp_dir_path, '1k_10k')),
    '10k_100k': os.listdir(os.path.join(tmp_dir_path, '10k_100k')),
    '100k_1m': os.listdir(os.path.join(tmp_dir_path, '100k_1m')),
    '1m_10m': os.listdir(os.path.join(tmp_dir_path, '1m_10m')),
}


def check_level(dealer_name):
    for k, v in d_dealers.items():
        if dealer_name in v:
            return k
    return ''


print('traversing data ... ')

file_list = os.listdir(dir_path)
length = len(file_list)
for i, dealer_dir_name in enumerate(file_list):
    if i % 2 == 0:
        progress = float(i + 1) / length * 100.
        print('\rprogress: %.2f%% ' % progress, end='')

    dealer_index = dealer_dir_name.split('_')[1]
    if dealer_index == '0':
        continue

    no_below_list = [1, 5, 10, 20, 30, 40]
    size_times = 1
    force_no_direction = False

    level = check_level(dealer_dir_name)
    if not level:
        continue

    tmp_save_dir = os.path.join(save_dir_path, level)
    if not os.path.isdir(tmp_save_dir):
        os.mkdir(tmp_save_dir)

    sub_dir_path = os.path.join(dir_path, dealer_dir_name)

    data = gen_inputs.__load_dir(sub_dir_path)
    dates, voc_size, original_bond_size = gen_inputs.__process(data, remove_bond_list, dir_name, 0,
                                                               force_no_direction)
    if voc_size == 0 or original_bond_size < 10:
        continue
    gen_inputs.plot(dates, voc_size, original_bond_size, name, tmp_save_dir, dealer_index, size_times, 0)

print('\nfinish traversing')

print('\ndone')
