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


def load_l_dealers(_suffix='date', use_cache=True):
    print('\nloading data ... ')

    cache_path = utils.get_relative_dir('runtime', 'cache', f'l_dealers_split_by_{_suffix}.pkl')
    if os.path.exists(cache_path) and use_cache:
        return utils.load_pkl(cache_path)

    # load d_dealers
    _d_dealers = utils.load_json(os.path.join(path.D_DEALERS_TRACE_DIR, f'train_{d_dealer_file_name}'))
    if '0' in _d_dealers:
        del _d_dealers['0']
    if '99999' in _d_dealers:
        del _d_dealers['99999']

    _l_dealers = []
    length = len(_d_dealers)
    count = 0

    for _dealer_index, traces in _d_dealers.items():
        # output progress
        count += 1
        if count % 2 == 0:
            progress = float(count) / length * 100.
            print('\rprogress: %.2f%% ' % progress, end='')

        # calculate the total transaction count
        tmp_trace_count = len(traces)

        # filter bonds whose trade frequency is low
        if tmp_trace_count < 1000:
            continue

        # calculate total volume of this dealer
        tmp_volume = np.sum(list(map(lambda x: x[1], traces)))

        # set filter boundaries according to the level of transaction count
        if tmp_trace_count > 100000:
            no_below = 45
        elif tmp_trace_count > 10000:
            no_below = 22
        else:
            no_below = 8

        # arrange the bonds according to their dates of transaction, for filtering purpose
        d_date_2_bonds = {}
        for v in traces:
            bond_id = v[0]
            trade_date = v[-1]

            if trade_date not in d_date_2_bonds:
                d_date_2_bonds[trade_date] = []
            d_date_2_bonds[trade_date].append(bond_id)

        # construct doc list for transactions
        tmp_doc_list = list(map(lambda x: x[1], d_date_2_bonds.items()))
        tmp_bonds = list(map(lambda x: x[0], traces))

        # construct dictionary for bonds
        tmp_dictionary = corpora.Dictionary(tmp_doc_list)
        num_bonds = len(tmp_dictionary)
        # filter bonds whose trade freq is low
        tmp_dictionary.filter_extremes(no_below=no_below, no_above=1., keep_n=2000)
        # num of bonds after filtering
        no_below_num_bonds = len(tmp_dictionary)

        # calculate the total transaction count and total volume after filtering
        tmp_bond_indices = tmp_dictionary.doc2idx(tmp_bonds)
        no_below_trace_count = 0
        no_below_volume = 0

        for i, v in enumerate(tmp_bond_indices):
            # if the bond not in the dictionary, then continue
            if v == -1:
                continue
            no_below_trace_count += 1
            no_below_volume += traces[i][1]

        # filter dealers whose total transaction count is low or number of unique bonds is low after filtering
        if no_below_trace_count == 0 or no_below_num_bonds <= 5:
            continue

        # get the unique bonds
        bond_set = set(list(tmp_dictionary.values()))

        # save features that is needed for clustering
        new_l = [_dealer_index, tmp_trace_count, tmp_volume, num_bonds, no_below_trace_count, no_below_volume,
                 no_below_num_bonds, bond_set]

        _l_dealers.append(new_l)

    # sort by total transaction count
    _l_dealers.sort(key=lambda x: -x[1])

    # cache data
    utils.write_pkl(cache_path, _l_dealers)
    return _l_dealers


def __generate_date_structure(len_bonds):
    """
    generate a empty zero matrix, shape: (dates, len_bonds)
    """
    start_date = '2015-01-02'
    end_date = '2015-10-21'
    start_timestamp = utils.date_2_timestamp(start_date)
    end_timestamp = utils.date_2_timestamp(end_date, True)

    l = []
    d = {}
    cur_timestamp = start_timestamp
    while cur_timestamp <= end_timestamp:
        _date = utils.timestamp_2_date(cur_timestamp)
        cur_timestamp += 86400

        if date.is_holiday(_date):
            continue

        d[_date] = len(l)
        l.append(np.zeros(len_bonds, ))

    return np.array(l), d


def __add_input_matrix_2_l_dealers(_l_dealers, _suffix='date', use_cache=True):
    cache_path = utils.get_relative_dir('runtime', 'cache', f'l_dealers_with_input_matrix_split_by_{_suffix}.pkl')
    if os.path.exists(cache_path) and use_cache:
        return utils.load_pkl(cache_path)

    # load d_dealers
    _d_dealers = utils.load_json(os.path.join(path.D_DEALERS_TRACE_DIR, f'train_{d_dealer_file_name}'))
    if '0' in _d_dealers:
        del _d_dealers['0']
    if '99999' in _d_dealers:
        del _d_dealers['99999']

    # generate dictionary for all dealers
    bond_sets = list(map(lambda x: list(x[-1]), _l_dealers))
    total_dictionary = corpora.Dictionary(bond_sets)
    len_bonds = len(total_dictionary)

    print(f'total bonds: {len_bonds}')

    # generate empty zero date matrix
    empty_inputs, dict_date_2_input_index = __generate_date_structure(len_bonds)

    d_dealer_index_2_input = {}

    for _dealer_index, val in _d_dealers.items():
        # generate a temp input matrix, so that empty matrix would not be override
        tmp_inputs = copy.deepcopy(empty_inputs)

        for v in val:
            bond_id = v[0]
            tmp_date = v[-1]

            # find bond index according to dictionary
            _idx = total_dictionary.doc2idx([bond_id])[0]

            # if bond is not in total dictionary or date is not weekdays
            if tmp_date not in dict_date_2_input_index or _idx == -1:
                continue

            # if there are transaction, then the corresponding bond is record to have transactions
            tmp_inputs[dict_date_2_input_index[tmp_date]][_idx] = 1

        d_dealer_index_2_input[_dealer_index] = tmp_inputs

    # add this input matrix to the last elements of each elements in origin_l_dealers
    for v in _l_dealers:
        _dealer_index = v[0]
        v.append(d_dealer_index_2_input[_dealer_index])

    utils.write_pkl(cache_path, _l_dealers)
    return _l_dealers


def __similarity(_points, use_bond_overlap=True, use_pattens=True):
    len_points = len(_points)

    # get statistics features
    stats = np.array(list(map(lambda x: list(x), _points[:, :-2])))

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

    # if only use basic features
    if not use_bond_overlap and not use_pattens:
        return sim_ret_1

    print('\tcalculating sim ret 2 ...')
    sim_ret_2 = np.zeros((len_points, len_points))
    bond_sets = _points[:, -2]
    for i, bonds_i in enumerate(bond_sets):
        for j, bonds_j in enumerate(bond_sets):
            intersec_num = len(bonds_i.intersection(bonds_j))
            sim_ret_2[i, j] = float(intersec_num) / max(len(bonds_i), len(bonds_j))

    # if use basic statistics features plus bond overlap info
    if not use_pattens:
        return 0.7 * sim_ret_1 + 0.3 * sim_ret_2

    print('\tcalculating sim ret 3 ...')
    sim_ret_3 = np.zeros((len_points, len_points))
    matrix_list = points[:, -1]
    for i, inputs_i in enumerate(matrix_list):
        if i % 2 == 0:
            progress = float(i + 1) / len_points * 100.
            print('\rprogress: %.2f%% ' % progress, end='')

        for j, inputs_j in enumerate(matrix_list):
            # sim_ret_3[i, j] = jaccard_similarity_score(inputs_i, inputs_j)
            # sim_ret_3[i, j] = np.mean(inputs_i * inputs_j)
            sim_ret_3[i, j] = np.mean((inputs_i == 0) * (inputs_j == 0) + inputs_i * inputs_j)

    return 0.4 * sim_ret_1 + 0.2 * sim_ret_2 + 0.4 * sim_ret_3


if __name__ == '__main__':
    suffix = 'date'
    d_dealer_file_name = f'd_dealers_2015_split_by_{suffix}.json'

    group_type = f'k_means_cluster_6_split_by_{suffix}'
    cluster_num = 6
    use_k_means = True
    use_spectral_clustering = False
    use_bond_overlap = False
    use_pattern_info = False

    # generate features for each dealer
    origin_l_dealers = load_l_dealers(_suffix=suffix, use_cache=True)
    origin_l_dealers = __add_input_matrix_2_l_dealers(origin_l_dealers, suffix, use_cache=True)

    print(f'\nnumber of dealers after filtering: {len(origin_l_dealers)}')

    print('\nConverting ...')

    # get the required statistics data
    origin_l_dealers = origin_l_dealers[:240]
    print('\nTake the first 240 dealers ... ')

    l_dealers = list(map(lambda x:
                         # [x[0], np.log(x[4]), np.log10(x[5] + 1.1), x[6], x[7], x[8]],
                         [x[0], np.log(x[1]), np.log10(x[2]), x[3], np.log(x[4]), np.log10(x[5] + 1.1), x[6], x[7],
                          x[8]],
                         origin_l_dealers))

    # l_dealers = list(map(lambda x: [x[0], x[1], np.log10(x[2]), x[3], x[4], np.log10(x[5] + 1.1), x[6], x[7]], l_dealers))

    print('Normalizing ...')

    # normalize
    points = np.array(list(map(lambda x: x[1:-2], l_dealers)))
    points = (points - np.mean(points, axis=0)) / np.std(points, axis=0)
    points = points / (np.max(points, axis=0) - np.min(points, axis=0))

    l_dealers = np.array(l_dealers)

    print('Clustering ...')

    # clustering
    if use_spectral_clustering:
        if not use_bond_overlap and not use_pattern_info:
            ret = SpectralClustering(n_clusters=cluster_num).fit(points)
            labels = ret.labels_

        else:
            A = __similarity(l_dealers[:, 1:], use_bond_overlap=use_bond_overlap, use_pattens=use_pattern_info)
            labels = spectral_clustering(A, n_clusters=cluster_num)
    else:
        ret = KMeans(n_clusters=cluster_num).fit(points)
        labels = ret.labels_

    print('Reducing dimensions ...')

    d_dealers_2_group = {}

    # assign label for each dealers
    for i, v in enumerate(points):
        dealer_index = origin_l_dealers[i][0]
        d_dealers_2_group[dealer_index] = int(labels[i])
        print(f'label: {labels[i]}, points: {origin_l_dealers[i][:-2]}')

    # save group information to file
    utils.write_json(utils.get_relative_dir('groups', f'group_{group_type}.json'), d_dealers_2_group)

    # Dimension reduction so that it can be visualized
    visual_points = ReduceDim.tsne(points, len(points), n_components=2)
    # visual_points = ReduceDim.pca(points, 2)
    # visual_points = np.array(list(map(lambda x: x[1:-2], origin_l_dealers)))[:, [0, 5]]

    print('Plotting ...')

    # visualization
    X, Y = list(zip(*visual_points))
    Visual.spots(X, Y, labels, f'{group_type} group dealers (t-sne visualization)',
                 spot_size=3, save_path=utils.get_relative_dir('groups', f'group_{group_type}.png'))
