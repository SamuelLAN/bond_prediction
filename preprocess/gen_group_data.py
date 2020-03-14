import os
from config import path
from lib import utils


def gen_group_according_to(file_path):
    print('loading data ...')
    dict_dealer_index_2_group = utils.load_json(file_path)

    data, d_dealers, total_volume, total_transaction_count, bound_timestamp, d_new_bonds = utils.load_pkl(
        os.path.join(path.ROOT_DIR, 'runtime', 'tmp123.pkl'))

    utils.write_pkl(os.path.join(path.ROOT_DIR, 'runtime', 'tmp_d_dealers.pkl'), d_dealers)
    # d_dealers = utils.load_pkl(os.path.join(path.ROOT_DIR, 'runtime', 'tmp_d_dealers.pkl'))

    labels = set(list(map(lambda x: x[1], dict_dealer_index_2_group.items())))
    group_list = [{} for i in range(len(labels))]

    print('traversing data ...')

    length = len(d_dealers)
    cur = 0
    for dealer_index, trace_list in d_dealers.items():
        # show progress
        if cur % 5 == 0:
            progress = float(cur + 1) / length * 100.
            print('\rprogress: %.2f%% ' % progress, end='')
        cur += 1

        if dealer_index not in dict_dealer_index_2_group:
            continue

        group_index = dict_dealer_index_2_group[dealer_index]
        group_list[group_index][dealer_index] = trace_list

    print('\rprogress: 100.0%  \nsaving data ...')

    plan_name = os.path.splitext(os.path.split(file_path)[1])[0] + '.json'
    group_path = os.path.join(path.DATA_ROOT_DIR, 'groups', plan_name)
    utils.write_json(group_path, group_list)


# gen_group_according_to(os.path.join(
#     path.ROOT_DIR,
#     'group',
#     # 'group_K-means_without_original_stat.json'
#     # 'group_Spectral_Clustering_without_original_stat_with_model_input_features.json'
#     # 'group_K-means_filter_lower_5.json'
#     'group_Spectral_Clustering_filter_lower_5_with_model_input_features.json'
# ))
#
# print('done')

_path = r'D:\Data\share_mine_laptop\community_detection\data\groups\group_Spectral_Clustering_filter_lower_5_with_model_input_features.json'
data = utils.load_json(_path)

for i, v in enumerate(data):
    print(i, len(v))
