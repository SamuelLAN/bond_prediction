import os
import copy
import numpy as np
from gensim import corpora
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import spectral_clustering, KMeans, SpectralClustering
from lib.ml import Visual, ReduceDim
from config import path, date
from lib import utils

# path_pkl_2015 = os.path.join(path.TRACE_DIR, 'finra_trace_2015.pkl')
# data = utils.load_pkl(path_pkl_2015)
#
# print('\nstart converting data ...')
#
# data = np.array(data)
# data = list(map(lambda x: {
#     'bond_id': x[0],
#     'offering_date': x[15],
#     'report_dealer_index': int(x[10]),
#     'contra_party_index': int(x[11]),
#     'date': x[9],
#     'volume': float(x[3]),
# }, data))
#
# print('finish converting\n\nstart traversing data ...')
#
# d_new_bonds = {}
# d_dealers = {}
#
# total_volume = 0.
# total_transaction_count = len(data)
#
# bound_timestamp = utils.date_2_timestamp('2014-06-01')
#
# length = len(data)
# for i, v in enumerate(data):
#     if i % 20 == 0:
#         progress = float(i) / length * 100.
#         print('\rprogress: %.2f%% ' % progress, end='')
#
#     bond_id = v['bond_id']
#     offering_date = v['offering_date']
#     report_dealer_index = str(v['report_dealer_index'])
#     contra_party_index = str(v['contra_party_index'])
#     date = v['date']
#     volume = v['volume']
#
#     if str(offering_date)[0] != '2':
#         continue
#
#     date_clean = str(date).split(' ')[0]
#     offering_date_clean = str(offering_date).split(' ')[0]
#     offering_timestamp = utils.date_2_timestamp(offering_date_clean)
#     if offering_timestamp >= bound_timestamp:
#         continue
#
#     trade_type = ''
#     if report_dealer_index == 0:
#         trade_type = 'BfC'
#
#         if contra_party_index not in d_dealers:
#             d_dealers[contra_party_index] = []
#         d_dealers[contra_party_index].append([bond_id, volume, trade_type, date_clean])
#
#     else:
#         if contra_party_index == 99999:
#             trade_type = 'StC'
#
#             if report_dealer_index not in d_dealers:
#                 d_dealers[report_dealer_index] = []
#             d_dealers[report_dealer_index].append([bond_id, volume, trade_type, date_clean])
#
#         else:
#             trade_type = 'DtD'
#
#             if report_dealer_index not in d_dealers:
#                 d_dealers[report_dealer_index] = []
#             d_dealers[report_dealer_index].append([bond_id, volume, trade_type, date_clean])
#
#             if contra_party_index not in d_dealers and contra_party_index != '0':
#                 d_dealers[contra_party_index] = []
#             d_dealers[contra_party_index].append([bond_id, volume, trade_type, date_clean])
#
#     v['type'] = trade_type
#
#     if bond_id not in d_new_bonds:
#         d_new_bonds[bond_id] = []
#     d_new_bonds[bond_id].append([volume, trade_type])
#
#     total_volume += volume
#
# print(f'\ntotal_volume: {total_volume}')
# print(f'total_transaction_count: {total_transaction_count}')
# print(f'num of new bonds: {len(d_new_bonds)}')
#
# # l_bonds = []
# # for bond_id, l in d_new_bonds.items():
# #     l_bonds.append([bond_id, len(l), np.sum(list(map(lambda x: x[0], l)))])
# # l_bonds.sort(key=lambda x: -x[1])
#
# if '0' in d_dealers:
#     del d_dealers['0']
# if '99999' in d_dealers:
#     del d_dealers['99999']

from six.moves import cPickle as pickle

# with open(os.path.join(path.ROOT_DIR, 'runtime', 'tmp123.pkl'), 'wb') as f:
#     pickle.dump([data, d_dealers, total_volume, total_transaction_count, bound_timestamp, d_new_bonds], f)

# with open(os.path.join(path.ROOT_DIR, 'runtime', 'tmp123.pkl'), 'rb') as f:
#     data, d_dealers, total_volume, total_transaction_count, bound_timestamp, d_new_bonds = pickle.load(f)
#
# origin_l_dealers = []
# length = len(d_dealers)
# count = 0
# for dealer_index, l in d_dealers.items():
#     count += 1
#     if count % 2 == 0:
#         progress = float(count) / length * 100.
#         print('\rprogress: %.2f%% ' % progress, end='')
#
#     tmp_trace_count = len(l)
#     if tmp_trace_count < 1000:
#         continue
#
#     tmp_volume = np.sum(list(map(lambda x: x[1], l)))
#
#     if tmp_trace_count > 100000:
#         no_below = 50
#     elif tmp_trace_count > 10000:
#         no_below = 25
#     else:
#         no_below = 10
#
#     tmp_d_bonds = {}
#     for v in l:
#         bond_id = v[0]
#         offering_date = v[-1]
#         if offering_date not in tmp_d_bonds:
#             tmp_d_bonds[offering_date] = []
#         tmp_d_bonds[offering_date].append(bond_id)
#
#     tmp_doc_list = list(map(lambda x: x[1], tmp_d_bonds.items()))
#     tmp_bonds = list(map(lambda x: x[0], l))
#
#     tmp_dictionary = corpora.Dictionary(tmp_doc_list)
#     num_bonds = len(tmp_dictionary)
#     tmp_dictionary.filter_extremes(no_below=no_below, no_above=1., keep_n=2000)
#     no_below_num_bonds = len(tmp_dictionary)
#
#     tmp_bond_indices = tmp_dictionary.doc2idx(tmp_bonds)
#     no_below_trace_count = 0
#     no_below_volume = 0
#
#     for i, v in enumerate(tmp_bond_indices):
#         if v == -1:
#             continue
#         no_below_trace_count += 1
#         no_below_volume += l[i][1]
#
#     if no_below_trace_count == 0:
#         continue
#
#     bond_set = set(list(tmp_dictionary.values()))
#     new_l = [dealer_index, tmp_trace_count, tmp_volume, num_bonds, no_below_trace_count, no_below_volume,
#              no_below_num_bonds, bond_set]
#
#     origin_l_dealers.append(new_l)
#
# origin_l_dealers.sort(key=lambda x: -x[1])
#
# with open(os.path.join(path.ROOT_DIR, 'runtime', 'tmp_l_dealers.pkl'), 'wb') as f:
#     pickle.dump(origin_l_dealers, f)
#
# # with open(os.path.join(path.ROOT_DIR, 'runtime', 'tmp_l_dealers.pkl'), 'rb') as f:
# #     origin_l_dealers = pickle.load(f)
# #
# # ------------------------- for model input ------------------------------
#
# total_bond_set = set()
# for v in origin_l_dealers:
#     total_bond_set = total_bond_set.union(v[-1])
#
# tmp_doc = [[v for v in total_bond_set]]
# total_dictionary = corpora.Dictionary(tmp_doc)
# len_bonds = len(total_dictionary)
#
# print(f'total bonds: {len_bonds}')
#
#
# def __generate_date_structure():
#     start_date = '2015-01-02'
#     end_date = '2015-12-31'
#     start_timestamp = utils.date_2_timestamp(start_date)
#     end_timestamp = utils.date_2_timestamp(end_date, True)
#
#     l = []
#     d = {}
#     cur_timestamp = start_timestamp
#     while cur_timestamp <= end_timestamp:
#         _date = utils.timestamp_2_date(cur_timestamp)
#         cur_timestamp += 86400
#
#         if date.is_holiday(_date):
#             continue
#
#         d[_date] = len(l)
#         l.append(np.zeros(len_bonds, ))
#
#     return np.array(l), d
#
#
# empty_inputs, dict_date_2_input_index = __generate_date_structure()
#
# d_dealer_index_2_input = {}
#
# for dealer_index, val in d_dealers.items():
#     tmp_inputs = copy.deepcopy(empty_inputs)
#     for v in val:
#         bond_id = v[0]
#         tmp_date = v[-1]
#
#         _idx = total_dictionary.doc2idx([bond_id])[0]
#         if tmp_date not in dict_date_2_input_index or _idx == -1:
#             continue
#         tmp_inputs[dict_date_2_input_index[tmp_date]][_idx] = 1
#     d_dealer_index_2_input[dealer_index] = tmp_inputs
#
# for v in origin_l_dealers:
#     dealer_index = v[0]
#     v.append(d_dealer_index_2_input[dealer_index])

# --------------------------------

print('Loading variables ...')

# utils.write_pkl(os.path.join(path.ROOT_DIR, 'runtime', 'tmp_origin_l_dealers.pkl'), origin_l_dealers)
origin_l_dealers = utils.load_pkl(os.path.join(path.ROOT_DIR, 'runtime', 'tmp_origin_l_dealers.pkl'))

print('Converting ...')

new_l_dealers = []
for v in origin_l_dealers:
    if v[-3] <= 5:
        continue
    new_l_dealers.append(v)

origin_l_dealers = new_l_dealers[:240]
l_dealers = list(map(lambda x:
                     # [x[0], np.log(x[4]), np.log10(x[5] + 1.1), x[6], x[7], x[8]],
                     [x[0], np.log(x[1]), np.log10(x[2]), x[3], np.log(x[4]), np.log10(x[5] + 1.1), x[6], x[7], x[8]],
                     origin_l_dealers))


# l_dealers = list(map(lambda x: [x[0], x[1], np.log10(x[2]), x[3], x[4], np.log10(x[5] + 1.1), x[6], x[7]], l_dealers))


def __similarity(points):
    # ret = rbf_kernel(points)
    # ret = cosine_similarity(points)
    # ret = (0.1 * rbf_kernel(points) + 0.9 * cosine_similarity(points)) / 2.
    # return ret

    # _mean = np.mean(points, axis=0)

    len_points = len(points)
    stats = np.array(list(map(lambda x: list(x), points[:, :-2])))

    print('\tcalculating sim ret 1 ...')
    sim_ret_1 = np.zeros((len_points, len_points))
    for i, val_i in enumerate(stats):
        tmp = 3. * np.mean(np.power(stats - val_i, 2), axis=-1)
        tmp = 1 - np.tanh(tmp)
        sim_ret_1[i] = tmp
        # ret[i] = 1. / (np.sum(np.power(points - val_i, 2), axis=-1) + 1.)

        # tmp_points = np.sum(np.power(points - _mean, 2), axis=-1)
        # tmp_val = np.sum(np.power(val_i - _mean, 2), axis=-1)
        # ret[i] = 1. / (np.abs(tmp_points - tmp_val) + 1.)

    print('\tcalculating sim ret 2 ...')
    sim_ret_2 = np.zeros((len_points, len_points))
    bond_sets = points[:, -2]
    for i, bonds_i in enumerate(bond_sets):
        for j, bonds_j in enumerate(bond_sets):
            intersec_num = len(bonds_i.intersection(bonds_j))
            sim_ret_2[i, j] = float(intersec_num) / max(len(bonds_i), len(bonds_j))

    # print('\tcalculating sim ret 3 ...')
    # sim_ret_3 = np.zeros((len_points, len_points))
    # matrix_list = points[:, -1]
    # for i, inputs_i in enumerate(matrix_list):
    #     if i % 2 == 0:
    #         progress = float(i + 1) / len_points * 100.
    #         print('\rprogress: %.2f%% ' % progress, end='')
    #
    #     for j, inputs_j in enumerate(matrix_list):
    #         # sim_ret_3[i, j] = jaccard_similarity_score(inputs_i, inputs_j)
    #         # sim_ret_3[i, j] = np.mean(inputs_i * inputs_j)
    #         sim_ret_3[i, j] = np.mean((inputs_i == 0) * (inputs_j == 0) + inputs_i * inputs_j)

    # ret = 0.4 * sim_ret_1 + 0.2 * sim_ret_2 + 0.4 * sim_ret_3
    ret = 0.7 * sim_ret_1 + 0.3 * sim_ret_2
    return ret


print('Normalizing ...')

points = list(map(lambda x: x[1:-2], l_dealers))
points = (points - np.mean(points, axis=0)) / np.std(points, axis=0)
points = points / (np.max(points, axis=0) - np.min(points, axis=0))

l_dealers = np.array(l_dealers)
l_dealers[:, 1:-2] = points
new_points = l_dealers[:, 1:]

print('Calculating similarity ...')

# A = __similarity(new_points)

print('Clustering ...')

# labels = spectral_clustering(A, n_clusters=4)
# ret = SpectralClustering(n_clusters=4).fit(points)
ret = KMeans(n_clusters=4).fit(points)
labels = ret.labels_

print('Reducing dimensions ...')

d_dealers_2_group = {}

for i, v in enumerate(points):
    d_dealers_2_group[origin_l_dealers[i][0]] = int(labels[i])
    print(f'label: {labels[i]}, points: {origin_l_dealers[i][:-2]}')

group_type = 'L-means_filter_lower_5'
utils.write_json(os.path.join(path.ROOT_DIR, 'group', f'group_{group_type}.json'), d_dealers_2_group)

visual_points = ReduceDim.tsne(points, len(points), n_components=2)
# visual_points = ReduceDim.pca(points, 2)

print('Plotting ...')

X, Y = list(zip(*visual_points))
Visual.spots(X, Y, labels, f'{group_type} group dealers (t-sne visualization)',
             spot_size=3, save_path=f'D:\Github\\bond_prediction\group\group_{group_type}.png')
