import os
import numpy as np
from lib import utils
from config import path


def run(start_date, end_date, min_offering_date):
    """ 
    Load data from FINRA_TRACE pkl file into d_bond_id_2_trace 
    (key is bond id, value is a list of transaction history) 
    """

    # get cache path for train set and test set
    cache_name = f'd_bond_id_2_trace_with_start_{start_date}_end_{end_date}_offer_before_{min_offering_date}.json'
    cache_path = os.path.join(path.CACHE_DIR, cache_name)
    # load from cache if exists
    if os.path.exists(cache_path):
        return utils.load_json(cache_path)

    d_bond_id_2_trace = {}

    trace_dir = path.TRACE_DIR
    for file_name in os.listdir(trace_dir):
        print('loading data from {file_name} ...')

        # load data to memory
        file_path = os.path.join(trace_dir, file_name)
        tmp_data = utils.load_pkl(file_path)

        print('filtering data ...')
        tmp_data = tmp_data[tmp_data['TRD_RPT_DTTM'] >= start_date]
        tmp_data = tmp_data[tmp_data['TRD_RPT_DTTM'] <= end_date]
        tmp_data = tmp_data[tmp_data['OFFERING_DATE'] < min_offering_date]

        print('formatting data ...')
        tmp_data = list(map(lambda x: {
            'bond_id': str(x[0]),
            'report_dealer_index': str(x[10]),
            'contra_party_index': str(x[11]),
            'date': str(x[9]).strip(' ')[0],
            'volume': float(x[3]),
        }, np.array(tmp_data)))

        print('putting trace into d_bond_id_2_trace ...')
        for val in tmp_data:
            bond_id = val['bond_id']
            _date = val['date']

            if bond_id not in d_bond_id_2_trace:
                d_bond_id_2_trace[bond_id] = []
            d_bond_id_2_trace[bond_id].append(val)

    print('sorting trace according to date ...')
    for k, trace in d_bond_id_2_trace.items():
        trace.sort(key=lambda x: x['date'])

    print(f'caching result to d_bond_id_2_trace.json ...')
    utils.write_json(cache_path, d_bond_id_2_trace)

    print('finish loading data from FINRA TRACE pkl files \n')
    return d_bond_id_2_trace
