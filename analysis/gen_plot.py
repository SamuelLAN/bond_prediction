import os
from config import path
from lib import utils
from matplotlib import pyplot as plt

# load trace data
_trace_suffix = 'd_dealers_2015_split_by_date.json'
train_d_dealers = utils.load_json(os.path.join(path.D_DEALERS_TRACE_DIR, f'train_{_trace_suffix}'))
# test_d_dealers = utils.load_json(os.path.join(path.D_DEALERS_TRACE_DIR, f'test_{_trace_suffix}'))

cluster_path = utils.get_relative_file('groups', 'group_k_means_split_by_date.json')
cluster_json = utils.load_json(cluster_path)

labels_path = utils.get_relative_file('runtime', 'd_dealer_index_2_eco_label.json')
eco_label_dict = utils.load_json(labels_path)
eco_labels = []
cluster_labels = []

# assign label for each dealers
for dealer_index, _val in train_d_dealers.items():

    # if dealer_index not in cluster_json:
    #     continue

    _tmp_label_val = eco_label_dict[dealer_index] if dealer_index in eco_label_dict else 0
    if isinstance(_tmp_label_val, int):
        eco_labels.append('peri')
    else:
        eco_labels.append(_tmp_label_val)

    _tmp_label_val = cluster_json[dealer_index] if dealer_index in cluster_json else 4
    cluster_labels.append(_tmp_label_val)
    # print(cluster_json[dealer_index])

x = []
y = []

for dealer_index, trace_list in train_d_dealers.items():
    # print(dealer_index)

    # if dealer_index not in cluster_json:
    #     continue

    count_trace = len(trace_list)
    bonds = list(map(lambda x: x[0], trace_list))
    bonds = list(set(bonds))
    count_bond = len(bonds)

    # if count_trace >= 100000 or count_bond > 5000:
    #     continue

    x.append(count_trace)
    y.append(count_bond)

print(f'\nplotting... ')

# new_data = list(map(lambda x: [x[x_index], x[y_index]], data))
# dict_of_x = {}
#
# for v in new_data:
#     x = v[0]
#     if x not in dict_of_x:
#         dict_of_x[x] = []
#     dict_of_x[x].append(v[1])
#
# plot_data = list(map(lambda x: [x[0], np.mean(x[1])], dict_of_x.items()))
# plot_data.sort(key=lambda x: x[0])
#
# if not isinstance(remove_last_num, type(None)):
#     plot_data = plot_data[:-remove_last_num]
# X, Y = list(zip(*plot_data))

data = list(zip(x, y, eco_labels, cluster_labels))

print(f'num of spot: {len(data)}')

data_least = list(filter(lambda a: a[-1] == 3, data))
data_less = list(filter(lambda a: a[-1] == 0, data))
data_more = list(filter(lambda a: a[-1] == 1, data))
data_most = list(filter(lambda a: a[-1] == 2, data))
data_other = list(filter(lambda a: a[-1] == 4, data))

# data_least = list(filter(lambda a: a[0] < 5000, data))
# data_less = list(filter(lambda a: 5000 <= a[0] < 20000, data))
# data_more = list(filter(lambda a: 20000 <= a[0] < 100000, data))
# data_most = list(filter(lambda a: 100000 <= a[0], data))

data_other_core = list(filter(lambda a: a[-2] == 'core', data_other))
data_other_peri = list(filter(lambda a: a[-2] == 'peri', data_other))
data_other_idb = list(filter(lambda a: a[-2] == 'IDB', data_other))
data_other_unknown = list(filter(lambda a: a[-2] == 'unknown', data_other))

data_least_core = list(filter(lambda a: a[-2] == 'core', data_least))
data_least_peri = list(filter(lambda a: a[-2] == 'peri', data_least))
data_least_idb = list(filter(lambda a: a[-2] == 'IDB', data_least))
data_least_unknown = list(filter(lambda a: a[-2] == 'unknown', data_least))

data_less_core = list(filter(lambda a: a[-2] == 'core', data_less))
data_less_peri = list(filter(lambda a: a[-2] == 'peri', data_less))
data_less_idb = list(filter(lambda a: a[-2] == 'IDB', data_less))
data_less_unknown = list(filter(lambda a: a[-2] == 'unknown', data_less))

data_more_core = list(filter(lambda a: a[-2] == 'core', data_more))
data_more_peri = list(filter(lambda a: a[-2] == 'peri', data_more))
data_more_idb = list(filter(lambda a: a[-2] == 'IDB', data_more))
data_more_unknown = list(filter(lambda a: a[-2] == 'unknown', data_more))

data_most_core = list(filter(lambda a: a[-2] == 'core', data_most))
data_most_peri = list(filter(lambda a: a[-2] == 'peri', data_most))
data_most_idb = list(filter(lambda a: a[-2] == 'IDB', data_most))
data_most_unknown = list(filter(lambda a: a[-2] == 'unknown', data_most))

print('\n-------------------------------------------')
print(f'data_most: {len(data_most)}')
print(f'data_most_core: {len(data_most_core)}')
print(f'data_most_peri: {len(data_most_peri)}')
print(f'data_most_idb: {len(data_most_idb)}')
print(f'data_most_unknown: {len(data_most_unknown)}')

print('\n-------------------------------------------')
print(f'data_more: {len(data_more)}')
print(f'data_more_core: {len(data_more_core)}')
print(f'data_more_peri: {len(data_more_peri)}')
print(f'data_more_idb: {len(data_more_idb)}')
print(f'data_more_unknown: {len(data_more_unknown)}')

print('\n-------------------------------------------')
print(f'data_less: {len(data_less)}')
print(f'data_less_core: {len(data_less_core)}')
print(f'data_less_peri: {len(data_less_peri)}')
print(f'data_less_idb: {len(data_less_idb)}')
print(f'data_less_unknown: {len(data_less_unknown)}')

print('\n-------------------------------------------')
print(f'data_least: {len(data_least)}')
print(f'data_least_core: {len(data_least_core)}')
print(f'data_least_peri: {len(data_least_peri)}')
print(f'data_least_idb: {len(data_least_idb)}')
print(f'data_least_unknown: {len(data_least_unknown)}')

print('\n-------------------------------------------')
print(f'data_other: {len(data_other)}')
print(f'data_other_core: {len(data_other_core)}')
print(f'data_other_peri: {len(data_other_peri)}')
print(f'data_other_idb: {len(data_other_idb)}')
print(f'data_other_unknown: {len(data_other_unknown)}')

# exit()

plt.figure(figsize=(18., 18 * 4.8 / 10.4))

peri_marker = 'o'
idb_marker = 'x'
core_marker = 'd'

peri_size = 9
idb_size = 40
core_size = 40

x, y, z, _ = list(zip(*data_other_peri))
print(len(x))
plt.scatter(x, y, color='purple', marker=peri_marker, s=peri_size, label='outside cluster')

# x, y, z, _ = list(zip(*data_other_idb))
# print(len(x))
# plt.scatter(x, y, color='purple', marker=idb_marker, s=idb_size, label='outside cluster (IDB)')

x, y, z, _ = list(zip(*data_least_peri))
print(len(x))
plt.scatter(x, y, color='red', marker=peri_marker, s=peri_size, label='least active')

# x, y, z, _ = list(zip(*data_least_idb))
# print(len(x))
# plt.scatter(x, y, color='red', marker=idb_marker, s=idb_size, label='least active (IDB)')

x, y, z, _ = list(zip(*data_less_peri))
print(len(x))
plt.scatter(x, y, color='blue', marker=peri_marker, s=peri_size, label='less active')

# x, y, z, _ = list(zip(*data_less_idb))
# print(len(x))
# plt.scatter(x, y, color='blue', marker=idb_marker, s=idb_size, label='low active (IDB)')

x, y, z, _ = list(zip(*data_more_peri))
print(len(x))
plt.scatter(x, y, color='green', marker=peri_marker, s=peri_size, label='more active')

# x, y, z, _ = list(zip(*data_more_idb))
# print(len(x))
# plt.scatter(x, y, color='green', marker=idb_marker, s=idb_size, label='high active (IDB)')

# x, y, z, _ = list(zip(*data_more_core))
# print(len(x))
# plt.scatter(x, y, color='green', marker=core_marker, s=core_size, label='high active (core)')

x, y, z, _ = list(zip(*data_most_peri))
print(len(x))
plt.scatter(x, y, color='black', marker=peri_marker, s=peri_size, label='most active')

# x, y, z, _ = list(zip(*data_most_idb))
# print(len(x))
# plt.scatter(x, y, color='black', marker=idb_marker, s=idb_size, label='most active (IDB)')

# x, y, z, _ = list(zip(*data_most_core))
# print(len(x))
# plt.scatter(x, y, color='black', marker=core_marker, s=core_size, label='most active (core)')

# x, y = list(zip(*data_less))
# print(len(x))
# plt.scatter(x, y, color='blue', s=6, label='less active')
#
# x, y = list(zip(*data_more))
# print(len(x))
# plt.scatter(x, y, color='green', s=6, label='more active')
#
# x, y = list(zip(*data_most))
# print(len(x))
# plt.scatter(x, y, color='black', s=6, label='most active')

# plt.title(title)
plt.xlabel('Count of transactions', fontsize=30)
plt.ylabel('Distinct bond count', fontsize=30)
# plt.xticks(list(range(0, 700000, 100000)), fontsize=20)
# plt.xticks([1, 10000, 100000, 1000000], ['0', '1e4', '1e5', '1e6'], fontsize=20)
# plt.yticks(list(range(0, 7000, 1000)), fontsize=20)
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.xlim(1000, 350000)
# plt.ylim(50, 10000)
plt.xlim(100, 700000)
plt.ylim(10, 10000)
plt.legend(fontsize=20, loc='lower right')
file_path = os.path.join(path.ROOT_DIR, 'runtime', 'tmp',
                         'plot_of_all_data_without_eco_features_without_eco_labels_x_trace_count_y_count_distinct_bond_both_log_scale_no_filtering_axis.png')
plt.savefig(file_path, dpi=400)
plt.show()

print('finish plotting')
