import os
import numpy as np
from matplotlib import pyplot as plt
from config import path
from lib import utils

data = []

dir_path = os.path.join(path.PREDICTION_DIR, 'bonds_by_dealer', '2015')

save_dir_path = os.path.join(path.ROOT_DIR, 'runtime', 'hist_for_interval')
transaction_count_levels = ['100_1k', '1k_10k', '10k_100k', '100k_1m']
for dir_name in transaction_count_levels:
    new_dir_path = os.path.join(save_dir_path, dir_name)
    if not os.path.isdir(new_dir_path):
        os.mkdir(new_dir_path)

print('traversing data ... ')

file_list = os.listdir(dir_path)
length = len(file_list)
for i, file_name in enumerate(file_list):
    if i % 20 == 0:
        progress = float(i + 1) / length * 100.
        print('\rprogress: %.2f%% ' % progress, end='')

    dealer_index = str(file_name.split('.')[0]).split('_')[1]
    if dealer_index == '0':
        continue

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
    transaction_count = len(tmp_data)

    print(f'\n\ndealer: {dealer_index}')
    print(f'mean_interval: {mean_interval}')
    print(f'mean_count_divide_interval: {mean_count_divide_interval}')
    print(f'mean_count: {mean_count}')
    print(f'bond_size: {bond_size}')
    print(f'transaction_count: {transaction_count}')

    tmp_save_dir = ''
    if transaction_count < 100:
        print('continue because of small transaction count')
        continue
    elif transaction_count < 1000:
        tmp_save_dir = os.path.join(save_dir_path, '100_1k')
    elif transaction_count < 10000:
        tmp_save_dir = os.path.join(save_dir_path, '1k_10k')
    elif transaction_count < 100000:
        tmp_save_dir = os.path.join(save_dir_path, '10k_100k')
    elif transaction_count < 1000000:
        tmp_save_dir = os.path.join(save_dir_path, '100k_1m')

    print(f'tmp_save_dir: {os.path.split(tmp_save_dir)[1]}')

    plt.figure(figsize=(20., 20 * 4.8 / 10.4))
    bins = list(range(0, 320, 10))
    bins.insert(1, 1)
    hist, _ = np.histogram(intervals, bins=bins)
    max_y = int(np.max(hist))
    y_unit = int(np.ceil(max_y / 20.))
    max_y = int((np.ceil(max_y / float(y_unit)) + 1) * y_unit)
    plt.hist(intervals, bins=bins)
    plt.title(f'histogram of interval length of dealer {dealer_index}\nrange(0, 310, 10)')
    plt.xlabel('interval length')
    plt.ylabel('numbers')
    plt.xticks(bins)
    plt.yticks(np.arange(0, max_y, y_unit))
    img_name = f'dealer_{dealer_index}_hist_of_interval_length.png'
    file_path = os.path.join(tmp_save_dir, img_name)
    plt.savefig(file_path, dpi=300)
    # plt.show()
    plt.close()

    plt.figure(figsize=(20., 20 * 4.8 / 10.4))
    max_x = 1.0
    min_x = 0.0
    x_unit = 0.05
    bins = list(np.arange(min_x, max_x + x_unit, x_unit))
    bins.insert(1, 0.01)
    hist, _ = np.histogram(count_divide_intervals, bins=bins)
    max_y = int(np.max(hist))
    y_unit = int(np.ceil(max_y / 20.))
    max_y = int((np.ceil(max_y / float(y_unit)) + 1) * y_unit)
    plt.hist(count_divide_intervals, bins=bins)
    range_str = 'range(0, 1, 0.05)'
    plt.title(
        f'histogram of count_divide_intervals of dealer {dealer_index}\n{range_str}')
    plt.xlabel('count divide intervals')
    plt.ylabel('numbers')
    x_ticks = list(np.arange(min_x, max_x + x_unit, x_unit))
    plt.xticks(x_ticks)
    plt.yticks(np.arange(0, max_y, y_unit))
    img_name = f'dealer_{dealer_index}_hist_of_count_divide_intervals.png'
    file_path = os.path.join(tmp_save_dir, img_name)
    plt.savefig(file_path, dpi=300)
    # plt.show()
    plt.close()

    plt.figure(figsize=(20., 20 * 4.8 / 10.4))
    max_x = max(np.max(counts), 10)
    min_x = np.min(counts)
    x_unit = np.ceil((max_x - min_x) / 20.)
    max_x = min_x + x_unit * 21 if (max_x - min_x) > 19 * x_unit else max_x + x_unit
    bins = list(np.arange(min_x, max_x, x_unit))
    hist, _ = np.histogram(counts, bins=bins)
    max_y = int(np.max(hist))
    y_unit = int(np.ceil(max_y / 20.))
    max_y = int((np.ceil(max_y / float(y_unit)) + 1) * y_unit)
    plt.hist(counts, bins=bins)
    plt.title(
        f'histogram of count_of_dates_that_has_transaction_for_each_bond of dealer {dealer_index}\nrange({min_x}, {max_x - x_unit}, {x_unit})')
    plt.xlabel('count of dates that has transaction for each bond')
    plt.ylabel('numbers')
    plt.xticks(np.arange(min_x, max_x, x_unit))
    plt.yticks(np.arange(0, max_y, y_unit))
    img_name = f'dealer_{dealer_index}_hist_of_count_of_dates_that_has_transaction_for_each_bond.png'
    file_path = os.path.join(tmp_save_dir, img_name)
    plt.savefig(file_path, dpi=300)
    # plt.show()
    plt.close()

print('\nfinish traversing')

print('\ndone')
