import os
import sys

__cur_dir = os.path.split(os.path.abspath(__file__))[0]
__root_dir = os.path.split(__cur_dir)[0]
sys.path.append(__root_dir)

import numpy as np
from config import path
from lib import utils

# config
split_by_date = True
split_by_trace = False

use_cache = True

train_start_date = '2015-01-02'
# val_start_date = '2015-10-22'
val_start_date = '2015-11-25'
test_start_date = '2015-11-25'

train_ratio = 0.9
val_ratio = 0.0
test_ratio = 0.1

filter_new_bond_date = '2014-06-01'

prefix = 'd_bonds_2015'
if split_by_date:
    prefix += '_split_by_date'
elif split_by_trace:
    prefix += '_split_by_trace'


def load_data(_use_cache=True):
    """
    :return
        d_dealers (dict): {
            dealer_index (str): [ [bond_id, volume, trade_type, date], ... ],
            ...
        }
    """

    cache_path = utils.get_relative_dir('runtime', 'cache', 'dict_bonds_trace_2015.json')
    if _use_cache and os.path.exists(cache_path):
        return utils.load_json(cache_path)

    # load all finra trace data in 2015
    path_pkl_2015 = os.path.join(path.TRACE_DIR, 'finra_trace_2015.pkl')
    _data = utils.load_pkl(path_pkl_2015)

    print('formatting data ...')

    # formatting data
    _data = np.array(_data)
    _data = list(map(lambda x: {
        'bond_id': str(x[0]),
        'offering_date': str(x[15]),
        'report_dealer_index': str(x[10]),
        'contra_party_index': str(x[11]),
        'date': str(x[9]),
        'volume': float(x[3]),
    }, _data))

    print('converting data to d_bonds ... ')

    # convert data to dict
    _d_bonds = {}

    len_data = len(_data)
    for i, val in enumerate(_data):
        # output progress
        if i % 20 == 0:
            progress = float(i + 1) / len_data * 100.
            print('\rprogress: %.2f%% ' % progress, end='')

        # filter all transactions from new issued bonds
        offering_date = val['offering_date']
        if offering_date > filter_new_bond_date or offering_date[0] != '2':
            continue

        # read from dict
        bond_id = val['bond_id']
        report_dealer_index = val['report_dealer_index']
        contra_party_index = val['contra_party_index']
        date_clean = val['date'].split(' ')[0]
        volume = val['volume']

        if bond_id not in _d_bonds:
            _d_bonds[bond_id] = []
        _d_bonds[bond_id].append({
            'bond_id': bond_id,
            'volume': volume,
            'report_dealer_index': report_dealer_index,
            'contra_party_index': contra_party_index,
            'date': date_clean,
        })

    print('sorting data ...')

    # sort transactions according to dates
    for bond_id, trace in _d_bonds.items():
        trace.sort(key=lambda x: x['date'])

    print('finish all loading process\n')

    # cache data
    utils.write_json(cache_path, _d_bonds)
    return _d_bonds


print('\nloading data ...\n')

d_bonds = load_data(use_cache)

train_d_bonds = {}
val_d_bonds = {}
test_d_bonds = {}

print('\nsplitting data ...')

# split data according to the ratio of transactions
if split_by_trace:
    for bond_id, traces in d_bonds.items():
        # calculate indices
        len_trace = len(traces)
        val_start_index = int(len_trace * train_ratio)
        test_start_index = int(len_trace * (train_ratio + val_ratio))

        # split data
        train_d_bonds[bond_id] = traces[:val_start_index]
        val_d_bonds[bond_id] = traces[val_start_index:test_start_index]
        test_d_bonds[bond_id] = traces[test_start_index:]

# split data according to the date boundaries
elif split_by_date:
    for bond_id, traces in d_bonds.items():
        train_d_bonds[bond_id] = list(filter(lambda x: x['date'] < val_start_date, traces))
        val_d_bonds[bond_id] = list(filter(lambda x: val_start_date <= x['date'] < test_start_date, traces))
        test_d_bonds[bond_id] = list(filter(lambda x: test_start_date <= x['date'], traces))

print('saving data ...')

utils.write_json(os.path.join(path.D_BONDS_TRACE_DIR, f'train_{prefix}.json'), train_d_bonds)
utils.write_json(os.path.join(path.D_BONDS_TRACE_DIR, f'val_{prefix}.json'), val_d_bonds)
utils.write_json(os.path.join(path.D_BONDS_TRACE_DIR, f'test_{prefix}.json'), test_d_bonds)

print('done')
