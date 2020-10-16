import os
import numpy as np
from matplotlib import pyplot as plt
from config import path
from lib import utils

data = []

dir_path = os.path.join(path.PREDICTION_DIR, 'bonds_by_dealer', '2015')

print('reading data ... ')

check_balance_num = 0

distinct_count_of_bonds_for_fraction_of_dealers_below_half = []
distinct_count_of_bonds_for_fraction_of_dealers_over_half = []
distinct_count_of_bonds_for_dealers_transaction_count_0_2k = []
distinct_count_of_bonds_for_dealers_transaction_count_2k_10k = []
distinct_count_of_bonds_for_dealers_transaction_count_10k_100k = []
distinct_count_of_bonds_for_dealers_transaction_count_100k_1m = []

file_list = os.listdir(dir_path)
length = len(file_list)
for i, file_name in enumerate(file_list):
    if i % 20 == 0:
        progress = float(i + 1) / length * 100.
        print('\rprogress: %.2f%% ' % progress, end='')

    dealer_index = str(file_name.split('.')[0]).split('_')[1]

    tmp_data = utils.load_json(os.path.join(dir_path, file_name))

    d_bonds = {}
    for i, v in enumerate(tmp_data):
        bond_id = v[0]
        if bond_id not in d_bonds:
            d_bonds[bond_id] = []
        d_bonds[bond_id].append(v[2])

    l_bonds = []
    for bond_id, date_list in d_bonds.items():
        date_list.sort()
        first_date = date_list[0][:-9]
        last_date = date_list[-1][:-9]

        first_date_timestamp = utils.date_2_timestamp(first_date)
        last_date_timestamp = utils.date_2_timestamp(last_date)
        interval = int((last_date_timestamp - first_date_timestamp) / 86400)
        if interval > 0:
            interval += 1

        l_dates = list(map(lambda x: x[:-9], date_list))
        l_dates = list(set(l_dates))
        count = len(l_dates)
        if interval == 0:
            count = 0

        count_divide_interval = min(float(count) / interval, 1.0) if interval > 0 else 0

        l_bonds.append([bond_id, interval, count_divide_interval, count])

    intervals = list(map(lambda x: x[1], l_bonds))
    count_divide_intervals = list(map(lambda x: x[2], l_bonds))
    counts = list(map(lambda x: x[-1], l_bonds))

    mean_interval = np.mean(intervals)
    mean_count_divide_interval = np.mean(count_divide_intervals)
    mean_count = np.mean(counts)
    bond_size = len(d_bonds)

    bonds_no_below_5 = list(map(lambda x: x if x[-1] > 5 else '', l_bonds))
    while '' in bonds_no_below_5:
        bonds_no_below_5.remove('')
    bond_size_no_below_5 = len(bonds_no_below_5)

    bonds_no_below_10 = list(map(lambda x: x if x[-1] > 10 else '', l_bonds))
    while '' in bonds_no_below_10:
        bonds_no_below_10.remove('')
    bond_size_no_below_10 = len(bonds_no_below_10)

    transactions = list(map(lambda x: x[-1], tmp_data))

    transaction_count_of_all = len(transactions)
    transaction_count_of_BfC = transactions.count('bfc')
    transaction_count_of_StC = transactions.count('stc')
    transaction_count_of_BfD = transactions.count('bfd')
    transaction_count_of_StD = transactions.count('std')
    transaction_count_of_BfC_StC = transaction_count_of_BfC + transaction_count_of_StC
    transaction_count_of_BfD_StD = transaction_count_of_BfD + transaction_count_of_StD

    fraction_of_clients = transaction_count_of_BfC_StC / float(transaction_count_of_all)
    fraction_of_dealers = transaction_count_of_BfD_StD / float(transaction_count_of_all)

    facing = 'client-facing' if transaction_count_of_BfC_StC > transaction_count_of_BfD_StD else 'dealer-facing'
    dealer_type = ''
    if fraction_of_dealers >= 0.9:
        dealer_type = 'inter-dealers'
    elif 0.4 < fraction_of_dealers < 0.6:
        dealer_type = 'balance'
    elif fraction_of_clients > 0.9:
        dealer_type = 'basically-clients'

    bonds = list(map(lambda x: x[0], tmp_data))
    bonds = list(set(bonds))
    distinct_count_of_bonds = len(bonds)

    contra_dealers = list(map(lambda x: x[1], tmp_data))
    contra_dealers = list(set(contra_dealers))
    distinct_count_of_contra_dealers = len(contra_dealers)

    if 0.495 <= fraction_of_dealers <= 0.505:
        check_balance_num += 1

    data.append([
        dealer_index,
        transaction_count_of_all,
        transaction_count_of_BfC,
        transaction_count_of_StC,
        transaction_count_of_BfD,
        transaction_count_of_StD,
        transaction_count_of_BfC_StC,
        transaction_count_of_BfD_StD,
        fraction_of_clients,
        fraction_of_dealers,
        facing,
        dealer_type,
        distinct_count_of_bonds,
        distinct_count_of_contra_dealers,
        bond_size,
        bond_size_no_below_5,
        bond_size_no_below_10,
        mean_interval,
        mean_count_divide_interval,
        mean_count,
    ])

    if fraction_of_dealers < 0.49:
        distinct_count_of_bonds_for_fraction_of_dealers_below_half.append(distinct_count_of_bonds)
    elif fraction_of_dealers > 0.51:
        distinct_count_of_bonds_for_fraction_of_dealers_over_half.append(distinct_count_of_bonds)

    if 0 < transaction_count_of_all < 2000:
        distinct_count_of_bonds_for_dealers_transaction_count_0_2k.append(distinct_count_of_bonds)
    elif 2000 <= transaction_count_of_all < 10000:
        distinct_count_of_bonds_for_dealers_transaction_count_2k_10k.append(distinct_count_of_bonds)
    elif 10000 <= transaction_count_of_all < 100000:
        distinct_count_of_bonds_for_dealers_transaction_count_10k_100k.append(distinct_count_of_bonds)
    elif 100000 <= transaction_count_of_all < 1000000:
        distinct_count_of_bonds_for_dealers_transaction_count_100k_1m.append(distinct_count_of_bonds)

print('\nfinish reading\n\nsorting data ...')

# data.sort(key=lambda x: -x[1])
#
# print('\nconcating data to string ... ')
#
# string_data = list(map(lambda x: ','.join(list(map(lambda a: str(a), x))), data))
# string = 'dealer_index,all,BfC,StC,BfD,StD,BfC+StC,BfD+StD,fraction of clients,fraction of dealers,facing,' \
#          'dealer_type,distinct count of bonds,distinct count of contra dealers,' \
#          'bond size,bond size(count > 5),bond size(count > 10),' \
#          'mean interval,mean count_of_dates/interval,mean count of dates\n'
# string += '\n'.join(string_data)
#
# print('\nwriting data to file ...')
#
# with open(os.path.join(path.ROOT_DIR, 'runtime', 'stats_of_dealers.csv'), 'w') as f:
#     f.write(string)

# print(f'finish writing\n\ncheck_balance_num: {check_balance_num}')
# exit()

def generate_chart_for_distinct_count_of_bonds(x_index, title, x_label, y_label,
                                               remove_last_num=None, img_name=None, x_ticks=None, y_ticks=None,
                                               y_index=12):
    print(f'\nplotting {title} ... ')

    new_data = list(map(lambda x: [x[x_index], x[y_index]], data))
    dict_of_x = {}

    for v in new_data:
        x = v[0]
        if x not in dict_of_x:
            dict_of_x[x] = []
        dict_of_x[x].append(v[1])

    plot_data = list(map(lambda x: [x[0], np.mean(x[1])], dict_of_x.items()))
    plot_data.sort(key=lambda x: x[0])

    if not isinstance(remove_last_num, type(None)):
        plot_data = plot_data[:-remove_last_num]
    X, Y = list(zip(*plot_data))

    plt.figure(figsize=(18., 18 * 4.8 / 10.4))
    plt.scatter(X, Y, color='red', s=2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if not isinstance(x_ticks, type(None)):
        plt.xticks(x_ticks)
    if not isinstance(y_ticks, type(None)):
        plt.yticks(y_ticks)
    if not isinstance(img_name, type(None)):
        file_path = os.path.join(path.ROOT_DIR, 'runtime', 'plots', img_name)
        plt.savefig(file_path, dpi=200)
    plt.show()

    print('finish plotting')


# generate_chart_for_distinct_count_of_bonds(1,
#                                            'mean interval for transaction count',
#                                            'transaction count',
#                                            'mean interval',
#                                            3,
#                                            'plot_of_mean_interval_for_transaction_count.png',
#                                            np.arange(0, 450000, 50000),
#                                            np.arange(0, 220, 10),
#                                            -3)
#
# generate_chart_for_distinct_count_of_bonds(1,
#                                            'mean interval for transaction count',
#                                            'transaction count',
#                                            'mean interval',
#                                            25,
#                                            'plot_of_mean_interval_for_transaction_count_zoom.png',
#                                            np.arange(0, 105000, 5000),
#                                            np.arange(0, 210, 10),
#                                            -3)
#
# generate_chart_for_distinct_count_of_bonds(1,
#                                            'mean count_of_dates/interval for transaction count',
#                                            'transaction count',
#                                            'mean count_of_dates/interval',
#                                            3,
#                                            'plot_of_mean_count_of_dates_divide_interval_for_transaction_count.png',
#                                            np.arange(0, 450000, 50000),
#                                            np.arange(0, 0.36, 0.02),
#                                            -2)
#
# generate_chart_for_distinct_count_of_bonds(1,
#                                            'mean count_of_dates/interval for transaction count',
#                                            'transaction count',
#                                            'mean count_of_dates/interval',
#                                            25,
#                                            'plot_of_mean_count_of_dates_divide_interval_for_transaction_count_zoom.png',
#                                            np.arange(0, 105000, 5000),
#                                            np.arange(0, 0.36, 0.02),
#                                            -2)
#
# generate_chart_for_distinct_count_of_bonds(1,
#                                            'mean count_of_dates for transaction count',
#                                            'transaction count',
#                                            'mean count of dates that have transactions',
#                                            3,
#                                            'plot_of_mean_count_of_dates_that_have_transaction_for_transaction_count.png',
#                                            np.arange(0, 450000, 50000),
#                                            np.arange(0, 34, 2),
#                                            -1)
#
# generate_chart_for_distinct_count_of_bonds(1,
#                                            'mean count_of_dates for transaction count',
#                                            'transaction count',
#                                            'mean count of dates that have transactions',
#                                            25,
#                                            'plot_of_mean_count_of_dates_that_have_transaction_for_transaction_count_zoom.png',
#                                            np.arange(0, 105000, 5000),
#                                            np.arange(0, 23, 1),
#                                            -1)
#
# generate_chart_for_distinct_count_of_bonds(9,
#                                            'distinct count of bonds for fraction of dealers',
#                                            'fraction of dealers',
#                                            'distinct count of bonds',
#                                            None,
#                                            'plot_of_distinct_c_of_bonds_for_fraction_dealers.png',
#                                            np.arange(0, 1.05, 0.05),
#                                            np.arange(0, 11000, 1000))
#
generate_chart_for_distinct_count_of_bonds(1,
                                           'distinct count of bonds for transaction count',
                                           'transaction count',
                                           'distinct count of bonds',
                                           3,
                                           'plot_of_distinct_c_of_bonds_for_transaction_count.png',
                                           np.arange(0, 450000, 50000),
                                           np.arange(0, 11000, 1000))
#
# generate_chart_for_distinct_count_of_bonds(1,
#                                            'distinct count of bonds for transaction count',
#                                            'transaction count',
#                                            'distinct count of bonds',
#                                            25,
#                                            'plot_of_distinct_c_of_bonds_for_transaction_count_zoom.png',
#                                            np.arange(0, 105000, 5000),
#                                            np.arange(0, 6300, 300))


def generate_hist_for_distinct_count_of_bonds(_tmp_data, bins, title, img_name=None, x_ticks=None, y_ticks=None):
    print(f'\nplotting {title} ... ')

    plt.figure(figsize=(18., 18 * 4.8 / 10.4))
    plt.hist(_tmp_data, bins=bins)
    plt.title(title)
    plt.xlabel('distinct count of bonds')
    plt.ylabel('numbers')
    if not isinstance(x_ticks, type(None)):
        plt.xticks(x_ticks)
    if not isinstance(y_ticks, type(None)):
        plt.yticks(y_ticks)
    if not isinstance(img_name, type(None)):
        file_path = os.path.join(path.ROOT_DIR, 'runtime', 'histograms', img_name)
        plt.savefig(file_path, dpi=200)
    plt.show()

    print('finish plotting')


print('distinct_count_of_bonds_for_fraction_of_dealers_below_half',
      np.min(distinct_count_of_bonds_for_fraction_of_dealers_below_half),
      np.max(distinct_count_of_bonds_for_fraction_of_dealers_below_half))
print('distinct_count_of_bonds_for_fraction_of_dealers_over_half',
      np.min(distinct_count_of_bonds_for_fraction_of_dealers_over_half),
      np.max(distinct_count_of_bonds_for_fraction_of_dealers_over_half))
print('distinct_count_of_bonds_for_dealers_transaction_count_0_2k',
      np.min(distinct_count_of_bonds_for_dealers_transaction_count_0_2k),
      np.max(distinct_count_of_bonds_for_dealers_transaction_count_0_2k))
print('distinct_count_of_bonds_for_dealers_transaction_count_2k_10k',
      np.min(distinct_count_of_bonds_for_dealers_transaction_count_2k_10k),
      np.max(distinct_count_of_bonds_for_dealers_transaction_count_2k_10k))
print('distinct_count_of_bonds_for_dealers_transaction_count_10k_100k',
      np.min(distinct_count_of_bonds_for_dealers_transaction_count_10k_100k),
      np.max(distinct_count_of_bonds_for_dealers_transaction_count_10k_100k))
print('distinct_count_of_bonds_for_dealers_transaction_count_100k_1m',
      np.min(distinct_count_of_bonds_for_dealers_transaction_count_100k_1m),
      np.max(distinct_count_of_bonds_for_dealers_transaction_count_100k_1m))

generate_hist_for_distinct_count_of_bonds(distinct_count_of_bonds_for_fraction_of_dealers_below_half,
                                          list(range(0, 2100, 100)),
                                          'histogram of the distinct count of bonds\n' + \
                                          'for fraction of dealers < 0.49 (range(0, 2000, 100))',
                                          'hist_distinct_c_of_bonds_for_fraction_dealers_less_049.png',
                                          np.arange(0, 2100, 100),
                                          np.arange(0, 280, 20))

generate_hist_for_distinct_count_of_bonds(distinct_count_of_bonds_for_fraction_of_dealers_over_half,
                                          list(range(0, 2100, 100)),
                                          'histogram of the distinct count of bonds\n' + \
                                          'for fraction of dealers > 0.51 range(0, 2000, 100)',
                                          'hist_distinct_c_of_bonds_for_fraction_dealers_over_051.png',
                                          np.arange(0, 2100, 100),
                                          np.arange(0, 360, 20))

generate_hist_for_distinct_count_of_bonds(distinct_count_of_bonds_for_dealers_transaction_count_0_2k,
                                          list(range(0, 420, 20)),
                                          'histogram of the distinct count of bonds\n' + \
                                          'for 0 < transaction count < 2k, range(0, 400, 20)',
                                          'hist_distinct_c_of_bonds_for_transaction_count_0_2k.png',
                                          np.arange(0, 420, 20),
                                          np.arange(0, 600, 50))

generate_hist_for_distinct_count_of_bonds(distinct_count_of_bonds_for_dealers_transaction_count_2k_10k,
                                          list(range(0, 2100, 100)),
                                          'histogram of the distinct count of bonds\n' + \
                                          'for 2k <= transaction count < 10k, range(0, 2000, 100)',
                                          'hist_distinct_c_of_bonds_for_transaction_count_2k_10k.png',
                                          np.arange(0, 2100, 100))

generate_hist_for_distinct_count_of_bonds(distinct_count_of_bonds_for_dealers_transaction_count_10k_100k,
                                          list(range(600, 5850, 250)),
                                          'histogram of the distinct count of bonds\n' + \
                                          'for 10k <= transaction count < 100k, range(600, 5600, 250)',
                                          'hist_distinct_c_of_bonds_for_transaction_count_10k_100k.png',
                                          np.arange(600, 5850, 250),
                                          np.arange(0, 11, 1))

generate_hist_for_distinct_count_of_bonds(distinct_count_of_bonds_for_dealers_transaction_count_100k_1m,
                                          list(range(2900, 9500, 300)),
                                          'histogram of the distinct count of bonds\n' + \
                                          'for 100k <= transaction count < 1m, range(2900, 9200, 300)',
                                          'hist_distinct_c_of_bonds_for_transaction_count_100k_1m.png',
                                          np.arange(2900, 9500, 300),
                                          np.arange(0, 6, 1))

print('\ndone')
