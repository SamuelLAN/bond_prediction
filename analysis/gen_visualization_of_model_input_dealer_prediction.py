import os
import numpy as np
from matplotlib import pyplot as plt
from config import path, date, load
from lib import utils

# from analysis import gen_inputs_2 as gen_inputs

data = []

# name_map = {
#     'buy_client': 'BfC',
#     'sell_client': 'StC',
#     'buy_dealer': 'BfD',
#     'sell_dealer': 'StD',
#     'clients': 'BfC+StC',
#     'dealers': 'BfD+StD',
#     '': 'all',
# }

# dir_name = 'bonds_by_dealer'
# dir_path = os.path.join(path.PREDICTION_DATE_DIR, dir_name, '2015')
dir_path = load.group_path_dealer_prediction

# name = dir_name.replace('bonds_by_dealer_', '').replace('bonds_by_dealer', '')
# name = name_map[name]

save_dir_path = os.path.join(path.ROOT_DIR, 'runtime', 'visualization_of_model_inputs_dealer_prediction')
if not os.path.isdir(save_dir_path):
    os.mkdir(save_dir_path)

# transaction_count_levels = ['100_1k', '1k_10k', '10k_100k', '100k_1m']
# for _dir_name in transaction_count_levels:
#     new_dir_path = os.path.join(save_dir_path, _dir_name)
#     if not os.path.isdir(new_dir_path):
#         os.mkdir(new_dir_path)
#
# path_dict_bond_id_offering_date = os.path.join(path.DATA_ROOT_DIR, 'dict_bond_id_offering_date.json')
# dict_bond_id_offering_date = utils.load_json(path_dict_bond_id_offering_date)
# remove_bond_list = []
#
# bound_date = '2014-12-01'
# bound_timestamp = utils.date_2_timestamp(bound_date)
# for bond_id, offering_date in dict_bond_id_offering_date.items():
#     if offering_date[:2] == '19':
#         continue
#     elif offering_date[:1] != '2':
#         remove_bond_list.append(bond_id)
#         continue
#
#     offering_date = offering_date.replace('-00', '-01')
#
#     offering_timestamp = utils.date_2_timestamp(offering_date)
#     if offering_timestamp >= bound_timestamp:
#         remove_bond_list.append(bond_id)
#
# tmp_dir_path = os.path.join(path.PREDICTION_DATE_BY_VOLUME_DIR, dir_name, '2015')
# d_dealers = {
#     '100_1k': os.listdir(os.path.join(tmp_dir_path, '100_1k')),
#     '1k_10k': os.listdir(os.path.join(tmp_dir_path, '1k_10k')),
#     '10k_100k': os.listdir(os.path.join(tmp_dir_path, '10k_100k')),
#     '100k_1m': os.listdir(os.path.join(tmp_dir_path, '100k_1m')),
#     '1m_10m': os.listdir(os.path.join(tmp_dir_path, '1m_10m')),
# }

# def check_level(dealer_name):
#     for k, v in d_dealers.items():
#         if dealer_name in v:
#             return k
#     return ''


print('traversing data ... ')

file_list = os.listdir(dir_path)
file_list = list(filter(lambda x: 'train' in x, file_list))
length = len(file_list)

for i, file_name in enumerate(file_list):
    file_path = os.path.join(dir_path, file_name)

    # for i, dealer_dir_name in enumerate(file_list):
    if i % 10 == 0:
        progress = float(i + 1) / length * 100.
        print('\rprogress: %.2f%% ' % progress, end='')

    bond_id = os.path.splitext(file_name)[0].split('_')[1]

    X, Y = utils.load_pkl(file_path)

    visual_data = list(map(lambda x: x[1], X)) + list(X[-1][2:-1])

    visual_x = []
    visual_y = []
    values = []
    for j, trace_in_a_day in enumerate(visual_data):
        tmp_y = list(map(lambda x: x[0], np.argwhere(trace_in_a_day != 0)))

        visual_x += [j + 1] * len(tmp_y)
        visual_y += tmp_y
        values += [trace_in_a_day[pos] for pos in tmp_y]

    d_value_2_x_y = {}
    for j, v in enumerate(values):
        if v not in d_value_2_x_y:
            d_value_2_x_y[v] = {'x': [], 'y': []}
        d_value_2_x_y[v]['x'].append(visual_x[j])
        d_value_2_x_y[v]['y'].append(visual_y[j])

    plt.figure(figsize=(20., 20 * 4.8 / 10.4))
    j = 0
    for s, x_y_dict in d_value_2_x_y.items():
        if j == 0:
            plt.scatter(x_y_dict['x'], x_y_dict['y'], color='red', s=s, label='trade')
        else:
            plt.scatter(x_y_dict['x'], x_y_dict['y'], color='red', s=s)
        j += 1

    plt.title(f'Pattern Visualization of Bond {bond_id}', fontsize=24)
    plt.xlabel('dates (only weekdays from Jan 2nd to Dec 31th)', fontsize=22)
    plt.ylabel('dealer index', fontsize=20)

    # max_y, min_y, unit_y = cal_max_min_unit_for_hist(Y, 25)
    # max_x, min_x, unit_x = cal_max_min_unit_for_hist(X, 25)

    plt.yticks(np.arange(max(0, min(visual_y) - 5), max(visual_y) + 5,
                         int(int((max(visual_y) - min(visual_y) + 10) / 12 / 10) * 10)), fontsize=18)
    plt.xticks(np.arange(0, 240, 40), fontsize=18)
    plt.legend(fontsize=22)

    # save_path = os.path.join(save_dir, f'dealer_{dealer_index}_no_below_{no_below}.png')
    plt.savefig(utils.get_relative_file('runtime', 'visualization_dealer_prediction', load.group_file_name, f'bond_{bond_id}.png'), dpi=300)
    # plt.show()
    plt.close()
    # exit()

    # no_below_list = [1, 5, 10, 20, 30, 40]
    # size_times = 1
    # force_no_direction = False

    # level = check_level(dealer_dir_name)
    # tmp_save_dir = os.path.join(save_dir_path, level)
    # if not os.path.isdir(tmp_save_dir):
    #     os.mkdir(tmp_save_dir)

    # sub_dir_path = os.path.join(dir_path, dealer_dir_name)
    #
    # data = gen_inputs.__load_dir(sub_dir_path)
    # for no_below in no_below_list:
    #     if level in ['100k_1m', '10k_100k'] and no_below in [1, 5]:
    #         continue
    #     if level in ['100_1k'] and no_below in [10, 20, 30, 40]:
    #         continue
    #     if level in ['1k_10k'] and no_below in [30, 40]:
    #         continue
    #     dates, voc_size, original_bond_size = gen_inputs.__process(data, remove_bond_list, dir_name, no_below,
    #                                                                force_no_direction)
    #     if voc_size == 0 or original_bond_size < 10:
    #         continue
    #     gen_inputs.plot(dates, voc_size, original_bond_size, name, tmp_save_dir, dealer_index, size_times, no_below)


# def plot(dates, voc_size, original_bond_size, name, save_dir, dealer_index, size_times, no_below):
#     print('preparing plotting data ...\n')
#     d = {}
#     l = []
#     for i, v in enumerate(dates):
#         where = list(map(lambda x: [i, x[0]], np.argwhere(v != 0)))
#         val = list(map(lambda x: x[0], v[np.argwhere(v != 0)]))
#
#         l += where
#
#         for j, count in enumerate(val):
#             if str(count) not in d:
#                 d[str(count)] = []
#             d[str(count)].append(where[j])
#
#     if not d:
#         return
#
#     l = list(zip(*l))
#     # plt.scatter(l[0], l[1], color='blue', s=1)
#
#     print('plotting ... \n')
#
#     rets = {}
#
#     plt.figure(figsize=(20., 20 * 4.8 / 10.4))
#     X = []
#     Y = []
#
#     for k, v in d.items():
#         # if 'buy', then 'blue'; if 'sell', then 'red'
#         color = 'blue' if float(k) >= 0 else 'red'
#         _type = 'buy' if float(k) >= 0 else 'sell'
#
#         if name in ['StC', 'StD']:
#             color = 'red'
#             _type = 'sell'
#
#         s = int(abs(float(k))) * size_times
#         v = list(zip(*v))
#
#         if _type not in rets:
#             rets[_type] = True
#         else:
#             _type = None
#
#         X += v[0]
#         Y += v[1]
#
#         p = plt.scatter(v[0], v[1], color=color, s=s, label=_type)
#         # rets[_type] = p
#
#     # plt_list = [v for k, v in rets.items()]
#
#     if dict_bond_id_topics:
#         title = f'dealer_{dealer_index} of {name} (topic_size: {voc_size}, original_bond_size: {original_bond_size})'
#     else:
#         title = f'dealer_{dealer_index} of {name} (bond_size: {voc_size}, no_below: {no_below}, original_bond_size: {original_bond_size})'
#
#     plt.title(title)
#     plt.xlabel('dates (only weekdays from Jan 2nd to Dec 31th)')
#     plt.ylabel('bond index')
#
#     max_y, min_y, unit_y = cal_max_min_unit_for_hist(Y, 25)
#     max_x, min_x, unit_x = cal_max_min_unit_for_hist(X, 25)
#
#     plt.xticks(np.arange(min_x, max_x, unit_x))
#     plt.yticks(np.arange(min_y, max_y, unit_y))
#     plt.legend()
#
#     save_path = os.path.join(save_dir, f'dealer_{dealer_index}_no_below_{no_below}.png')
#     plt.savefig(save_path, dpi=300)
#     plt.show()
#     plt.close()


print('\nfinish traversing')

print('\ndone')
