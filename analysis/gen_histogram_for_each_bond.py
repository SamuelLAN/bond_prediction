import os
import numpy as np
from matplotlib import pyplot as plt
from config import path
from lib import utils

print('start loading data ... ')

path_pkl_2015 = os.path.join(path.TRACE_DIR, 'finra_trace_2015.pkl')
data = utils.load_pkl(path_pkl_2015)

print('\nstart converting data ...')

data = np.array(data)
data = list(map(lambda x: {
    'bond_id': x[0],
    'offering_date': x[15],
    'report_dealer_index': int(x[10]),
    'contra_party_index': int(x[11]),
    'date': x[9],
    'volume': float(x[3]),
}, data))

print('finish converting\n\nstart traversing data ...')

d_bonds = {}
d_all_dealers = {}

for i, v in enumerate(data):
    bond_id = v['bond_id']
    offering_date = v['offering_date']
    report_dealer_index = v['report_dealer_index']
    contra_party_index = v['contra_party_index']
    date = v['date']
    volume = v['volume']

    trade_type = ''
    if report_dealer_index == 0:
        trade_type = 'BfC'
    else:
        if contra_party_index == 99999:
            trade_type = 'StC'
        else:
            trade_type = 'DtD'
    v['type'] = trade_type

    if bond_id not in d_bonds:
        d_bonds[bond_id] = {
            'buy_from_dealers': {},
            'sell_to_dealers': {},
            'dealers': {},
            'all': [],
        }

    d_bonds[bond_id]['buy_from_dealers'][report_dealer_index] = True
    d_bonds[bond_id]['sell_to_dealers'][contra_party_index] = True
    d_bonds[bond_id]['dealers'][report_dealer_index] = True
    d_bonds[bond_id]['dealers'][contra_party_index] = True
    d_bonds[bond_id]['all'].append(v)

    if report_dealer_index not in d_all_dealers:
        d_all_dealers[report_dealer_index] = []
    if contra_party_index not in d_all_dealers:
        d_all_dealers[contra_party_index] = []

    d_all_dealers[report_dealer_index].append(volume)
    d_all_dealers[contra_party_index].append(volume)

print('finish traversing')

l_buy_from_dealers = []
l_sell_to_dealers = []
l_dealers = []
l_count_trade = []
l_count_trade_divide_dealers = []
l_count_trade_and_dealers = []
l_volumes_for_each_bond = []
l_volumes_for_each_dealer = []

print('\nhandling dict bonds ...')

c = 0
for bond_id, val in d_bonds.items():
    val['count_buy_from_dealers'] = len(val['buy_from_dealers'])
    val['count_sell_to_dealers'] = len(val['sell_to_dealers'])
    val['count_dealers'] = len(val['dealers'])
    val['count_trade'] = len(val['all'])

    volumes = list(map(lambda x: x['volume'], val['all']))
    log_sum_volumes = np.log10(np.sum(volumes))

    l_buy_from_dealers.append(val['count_buy_from_dealers'])
    l_sell_to_dealers.append(val['count_sell_to_dealers'])
    l_dealers.append(val['count_dealers'])
    l_count_trade.append(val['count_trade'])
    l_count_trade_divide_dealers.append(float(val['count_trade']) / val['count_dealers'])
    l_volumes_for_each_bond.append(log_sum_volumes)
    l_count_trade_and_dealers.append([val['count_trade'], val['count_dealers']])

for dealer_index, volumes in d_all_dealers.items():
    log_sum_volumes = np.log10(np.sum(volumes))
    l_volumes_for_each_dealer.append(log_sum_volumes)

print('\nploting ...')


def cal_max_min_unit_for_hist(hist, num_unit, unit_is_integer=True):
    _max = np.max(hist)
    _min = np.min(hist)
    unit = float(_max - _min) / num_unit
    if unit_is_integer:
        if unit > 1:
            unit = np.ceil(unit)
            _max = int(_max) + unit - int(_max) % unit
        else:
            unit = 1
            _max = int(_max) + 1
    else:
        pass

    return _max + unit, _min, unit


def histogram(_list, title, x_label, save_path, remove_outliers_num=0):
    plt.figure(figsize=(18., 18 * 4.8 / 10.4))

    _list.sort()

    hist, bins = np.histogram(_list, bins=25)

    if remove_outliers_num:
        len_list = len(_list)
        for _i in range(1, remove_outliers_num + 1):
            if hist[-_i] >= len_list * 0.001:
                break

            if hist[-_i]:
                _list = _list[:-hist[-_i]]

        hist, bins = np.histogram(_list, bins=25)

    max_y, min_y, unit_y = cal_max_min_unit_for_hist(hist, 25)
    max_x, min_x, unit_x = cal_max_min_unit_for_hist(_list, 25, False)

    plt.hist(_list, bins=bins)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('numbers')

    plt.xticks(np.arange(min_x, max_x, unit_x))
    plt.yticks(np.arange(min_y, max_y, unit_y))

    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


def plot(_list, title, x_label, y_label, save_path, remove_last_num=None):
    plt.figure(figsize=(18., 18 * 4.8 / 10.4))

    if remove_last_num:
        _list.sort(key=lambda x: x[1])
        _list = _list[:-remove_last_num]

    X, Y = list(zip(*_list))
    plt.scatter(X, Y, color='red', s=2)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    max_y, min_y, unit_y = cal_max_min_unit_for_hist(Y, 25)
    max_x, min_x, unit_x = cal_max_min_unit_for_hist(X, 25)

    plt.xticks(np.arange(min_x, max_x, unit_x))
    plt.yticks(np.arange(min_y, max_y, unit_y))

    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


save_dir = os.path.join(path.ROOT_DIR, 'runtime', 'histograms_for_each_bond')
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

histogram(l_buy_from_dealers,
          'histogram of distinct count of dealers in buy position for each bond',
          'distinct count of dealers in buy position',
          os.path.join(save_dir, 'hist_of_count_of_dealers_in_buy_for_each_bond.png'),
          6)

histogram(l_sell_to_dealers,
          'histogram of distinct count of dealers in sell position for each bond',
          'distinct count of dealers in sell position',
          os.path.join(save_dir, 'hist_of_count_of_dealers_in_sell_for_each_bond.png'),
          6)

histogram(l_dealers,
          'histogram of distinct count of dealers for each bond',
          'distinct count of dealers',
          os.path.join(save_dir, 'hist_of_count_of_dealers_for_each_bond.png'),
          6)

histogram(l_count_trade,
          'histogram of count of trade for each bond',
          'count of trade',
          os.path.join(save_dir, 'hist_of_count_of_trade_for_each_bond.png'),
          16)

histogram(l_count_trade_divide_dealers,
          'histogram of count_trade / distinct_count_dealers for each bond',
          'count_trade / distinct_count_dealers',
          os.path.join(save_dir, 'hist_of_count_trade_divide_distinct_count_dealers_for_each_bond.png'),
          6)

histogram(l_volumes_for_each_bond,
          'histogram of log_sum_volumes for each bond',
          'log_sum_volumes',
          os.path.join(save_dir, 'hist_of_log_sum_volumes_for_each_bond.png'))

histogram(l_volumes_for_each_dealer,
          'histogram of log_sum_volumes for each dealer',
          'log_sum_volumes',
          os.path.join(save_dir, 'hist_of_log_sum_volumes_for_each_dealer.png'))

plot(l_count_trade_and_dealers,
     'plot of count_trade - count_dealers (for each bond)',
     'count_trade',
     'distinct_count_dealers',
     os.path.join(save_dir, 'plot_of_count_trade_and_distinct_count_dealers_for_each_bond.png'))

plot(l_count_trade_and_dealers,
     'plot of count_trade - count_dealers (for each bond)',
     'count_trade',
     'distinct_count_dealers',
     os.path.join(save_dir, 'plot_of_count_trade_and_distinct_count_dealers_for_each_bond_zoomed.png'),
     5)

print('\ndone')
