import os
import numpy as np
from lib import utils
from config import path


def run(d_bond_id_2_trace, split_bound, start_date, end_date, min_offering_date):
    # get cache path for train set and test set
    train_cache_name = f'd_bond_id_2_trace_with_start_{start_date}_end_{split_bound}_' \
                       f'offer_before_{min_offering_date}.json'
    test_cache_name = f'd_bond_id_2_trace_with_start_{split_bound}_end_{end_date}_' \
                      f'offer_before_{min_offering_date}.json'
    train_cache_path = os.path.join(path.CACHE_DIR, train_cache_name)
    test_cache_path = os.path.join(path.CACHE_DIR, test_cache_name)

    # load from cache if exists
    if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
        return utils.load_json(train_cache_path), utils.load_json(test_cache_path)

    # initialize variables
    d_bond_id_2_trace_for_train = {}
    d_bond_id_2_trace_for_test = {}

    print('splitting data ...')
    for bond_id, traces in d_bond_id_2_trace.items():
        d_bond_id_2_trace_for_train[bond_id] = list(filter(lambda x: x['date'] < split_bound, traces))
        d_bond_id_2_trace_for_test[bond_id] = list(filter(lambda x: split_bound <= x['date'], traces))

    print(f'caching result to {train_cache_name} and {test_cache_name} ...')
    utils.write_json(train_cache_path, d_bond_id_2_trace_for_train)
    utils.write_json(test_cache_path, d_bond_id_2_trace_for_test)

    print('finish splitting data into train set and test set \n')
    return d_bond_id_2_trace_for_train, d_bond_id_2_trace_for_test
