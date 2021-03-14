import os
import numpy as np
from config import path
from lib import utils

train_start_date = '2015-01-02'
train_end_date = '2015-11-25'


def load_trace_pickle(_use_cache=True):
    cache_path = utils.get_relative_dir('runtime', 'cache', 'trace_pkl_2015.json')
    if _use_cache and os.path.exists(cache_path):
        return utils.load_json(cache_path)

    # load all finra trace data in 2015
    path_pkl_2015 = os.path.join(path.TRACE_DIR, 'finra_trace_2015.pkl')

    print(f'loading data from {path_pkl_2015} ...')
    _data = utils.load_pkl(path_pkl_2015)

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
    _data.sort(key=lambda x: x['date'])

    print('finish loading ')

    utils.write_json(cache_path, _data)
    return _data


def traverse_trace_pickle(fn, _dict, filter_new_bond_date='2014-06-01'):
    _data = load_trace_pickle()

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

        fn(i, val, _dict)


def load_dealer_2_history(_use_cache=True, filter_new_bond_date='2014-06-01'):
    cache_path = utils.get_relative_dir('runtime', 'cache', 'dict_dealers_trace_2015.json')
    if _use_cache and os.path.exists(cache_path):
        return utils.load_json(cache_path)

    def get_dealer_history(i, val, _dict):
        # read from dict
        bond_id = val['bond_id']
        report_dealer_index = val['report_dealer_index']
        contra_party_index = val['contra_party_index']
        date_clean = val['date'].split(' ')[0]
        volume = val['volume']

        # pass transaction data to d_dealers
        if report_dealer_index == '0':
            trade_type = 'BfC'

            if contra_party_index not in _dict:
                _dict[contra_party_index] = []
            _dict[contra_party_index].append([bond_id, volume, trade_type, date_clean])

        else:
            if contra_party_index == '99999':
                trade_type = 'StC'

                if report_dealer_index not in _dict:
                    _dict[report_dealer_index] = []
                _dict[report_dealer_index].append([bond_id, volume, trade_type, date_clean])

            else:
                trade_type = 'StD'

                if report_dealer_index not in _dict:
                    _dict[report_dealer_index] = []
                _dict[report_dealer_index].append([bond_id, volume, trade_type, date_clean])

                trade_type = 'BfD'

                if contra_party_index != '0':
                    if contra_party_index not in _dict:
                        _dict[contra_party_index] = []
                    _dict[contra_party_index].append([bond_id, volume, trade_type, date_clean])

    _d_dealers = {}
    print('start getting dealer history and converting data format ')
    traverse_trace_pickle(get_dealer_history, _d_dealers, filter_new_bond_date)
    print('finish getting dealer history and converting ')

    # cache data
    utils.write_json(cache_path, _d_dealers)
    return _d_dealers


def load_dealer_2_dealer(_use_cache=True, filter_new_bond_date='2014-06-01', keep_dealers={}):
    cache_path = utils.get_relative_dir('runtime', 'cache', 'dict_dealer_2_dealer_2015.json')
    if _use_cache and os.path.exists(cache_path):
        return utils.load_json(cache_path)

    def get_dealer_2_dealer(i, val, _dict):
        bond_id = val['bond_id']
        report_dealer_index = val['report_dealer_index']
        contra_party_index = val['contra_party_index']
        volume = val['volume']
        date_clean = val['date'].split(' ')[0]

        # only use the training data for analysis
        if date_clean >= train_end_date:
            return

        if keep_dealers and (report_dealer_index not in keep_dealers or contra_party_index not in keep_dealers):
            return

        if report_dealer_index not in _dict:
            _dict[report_dealer_index] = {}
        if contra_party_index not in _dict[report_dealer_index]:
            _dict[report_dealer_index][contra_party_index] = {'count': 0, 'volume': 0, 'bonds': {}}
        _dict[report_dealer_index][contra_party_index]['count'] += 1
        _dict[report_dealer_index][contra_party_index]['volume'] += volume
        if bond_id not in _dict[report_dealer_index][contra_party_index]['bonds']:
            _dict[report_dealer_index][contra_party_index]['bonds'][bond_id] = 0
        _dict[report_dealer_index][contra_party_index]['bonds'][bond_id] += 1

        if contra_party_index not in _dict:
            _dict[contra_party_index] = {}
        if report_dealer_index not in _dict[contra_party_index]:
            _dict[contra_party_index][report_dealer_index] = {'count': 0, 'volume': 0, 'bonds': {}}
        _dict[contra_party_index][report_dealer_index]['count'] += 1
        _dict[contra_party_index][report_dealer_index]['volume'] += volume
        if bond_id not in _dict[contra_party_index][report_dealer_index]['bonds']:
            _dict[contra_party_index][report_dealer_index]['bonds'][bond_id] = 0
        _dict[contra_party_index][report_dealer_index]['bonds'][bond_id] += 1

    _d_dealer_2_dealer = {}
    print('start getting dealer_2_dealer data ... ')
    traverse_trace_pickle(get_dealer_2_dealer, _d_dealer_2_dealer, filter_new_bond_date)
    print('finish getting dealer_2_dealer data ')

    utils.write_json(cache_path, _d_dealer_2_dealer)
    return _d_dealer_2_dealer


def load_train_dealer_2_history():
    return dict(map(
        lambda x: (
            x[0],
            list(filter(lambda v: train_start_date <= v[-1] < train_end_date, x[1]))
        ),
        load_dealer_2_history().items()
    ))


def load_test_dealer_2_history():
    return dict(map(
        lambda x: (
            x[0],
            list(filter(lambda v: train_end_date <= v[-1], x[1]))
        ),
        load_dealer_2_history().items()
    ))


def load_d_dealers_after_filtering():
    cache_path = utils.get_relative_dir('cache', 'filtered_d_dealers_2_history.pkl', root=path.RUNTIME_DIR)
    if os.path.exists(cache_path):
        return utils.load_pkl(cache_path)

    from gensim import corpora

    # load trace data
    _train_d_dealers = load_train_dealer_2_history()
    _test_d_dealers = load_test_dealer_2_history()

    # remove clients, we do not predict clients' behaviors
    if '0' in _train_d_dealers:
        del _train_d_dealers['0']
    if '99999' in _train_d_dealers:
        del _train_d_dealers['99999']
    if '0' in _test_d_dealers:
        del _test_d_dealers['0']
    if '99999' in _test_d_dealers:
        del _test_d_dealers['99999']

    train_l_dealers = list(map(lambda x: [x[0], len(x[1])], _train_d_dealers.items()))
    train_l_dealers.sort(key=lambda x: -x[1])
    top_dealers = dict(map(lambda x: (x[0], True), train_l_dealers[:260]))

    new_train_d_dealers = {}
    new_test_d_dealers = {}

    length = len(_train_d_dealers)
    count = 0

    for _dealer_index, train_traces in _train_d_dealers.items():
        # output progress
        count += 1
        if count % 2 == 0:
            progress = float(count) / length * 100.
            print('\rprogress: %.2f%% ' % progress, end='')

        test_traces = _test_d_dealers[_dealer_index]

        # calculate the total transaction count
        train_trace_count = len(train_traces)

        # filter bonds whose trade frequency is low
        if train_trace_count < 1000 or _dealer_index not in top_dealers:
            continue

        # set filter boundaries according to the level of transaction count
        if train_trace_count > 100000:
            no_below = 45
        elif train_trace_count > 10000:
            no_below = 22
        else:
            no_below = 8

        # arrange the bonds according to their dates of transaction, for filtering purpose
        d_train_date_2_bonds = {}
        for v in train_traces:
            bond_id = v[0]
            trade_date = v[-1]

            if trade_date not in d_train_date_2_bonds:
                d_train_date_2_bonds[trade_date] = []
            d_train_date_2_bonds[trade_date].append(bond_id)

        # construct doc list for transactions
        train_doc_list = list(map(lambda x: x[1], d_train_date_2_bonds.items()))

        # construct dictionary for bonds
        train_dictionary = corpora.Dictionary(train_doc_list)
        # filter bonds whose trade freq is low
        train_dictionary.filter_extremes(no_below=no_below, no_above=1., keep_n=2000)
        # num of bonds after filtering
        no_below_num_bonds = len(train_dictionary)

        if no_below_num_bonds <= 5:
            continue

        # filter all traces whose bonds are not in the dictionary
        train_traces = list(filter(lambda x: train_dictionary.doc2idx([x[0]])[0] != -1, train_traces))
        test_traces = list(filter(lambda x: train_dictionary.doc2idx([x[0]])[0] != -1, test_traces))

        # filter this dealer if the transaction count of its filtered traces is small
        if len(train_traces) < 100 or len(test_traces) < 20:
            continue

        new_train_d_dealers[_dealer_index] = train_traces
        new_test_d_dealers[_dealer_index] = test_traces

    utils.write_pkl(cache_path, (new_train_d_dealers, new_test_d_dealers))
    return new_train_d_dealers, new_test_d_dealers


# train_d_dealers, test_d_dealers = load_d_dealers_after_filtering()
# print(f'len train_d_dealers: {len(train_d_dealers)}')
# print(f'len test_d_dealers: {len(test_d_dealers)}')
#
# train_d_dealers_2_dealers = load_dealer_2_dealer(False, keep_dealers=train_d_dealers)
# utils.write_json(utils.get_relative_file('runtime', 'cache', 'train_d_dealers_2_dealers_2015_filtered.json'),
#                  train_d_dealers_2_dealers)
# print(len(train_d_dealers_2_dealers))
