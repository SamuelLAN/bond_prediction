import os
import numpy as np
from gensim import corpora
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import spectral_clustering, KMeans, SpectralClustering
from config import path
from lib import utils
from lib.ml import Visual, ReduceDim


def load_l_dealers(use_cache=True):
    print('\nloading data ... ')

    cache_path = utils.get_relative_dir('runtime', 'cache', f'l_bonds.pkl')
    if os.path.exists(cache_path) and use_cache:
        return utils.load_pkl(cache_path)

    # load d_dealers
    _d_bonds = utils.load_json(os.path.join(path.D_BONDS_TRACE_DIR, f'train_d_bonds_2015_split_by_date.json'))

    _l_bonds = []
    length = len(_d_bonds)
    count = 0

    skip_count = 0
    for _bond_id, traces in _d_bonds.items():
        # output progress
        count += 1
        if count % 2 == 0:
            progress = float(count) / length * 100.
            print('\rprogress: %.2f%% ' % progress, end='')

        # calculate the total transaction count
        tmp_trace_count = len(traces)

        # filter bonds whose trade frequency is low
        # if tmp_trace_count < 1000:
        #     skip_count += 1
        #     continue

        # calculate total volume of this dealer
        tmp_volume = np.sum(list(map(lambda x: x['volume'], traces)))

        # TODO: check if we need to filter the dealer that only do very few transactions
        # # set filter boundaries according to the level of transaction count
        # if tmp_trace_count > 100000:
        #     no_below = 45
        # elif tmp_trace_count > 10000:
        #     no_below = 22
        # else:
        #     no_below = 8

        # TODO

        # arrange the bonds according to their dates of transaction, for filtering purpose
        d_date_2_dealers = {}
        for v in traces:
            report_dealer_id = v['report_dealer_index']
            contra_dealer_id = v['contra_party_index']
            trade_date = v['date']

            if trade_date not in d_date_2_dealers:
                d_date_2_dealers[trade_date] = []

            if report_dealer_id not in ['0', '99999']:
                d_date_2_dealers[trade_date].append(report_dealer_id)

            if contra_dealer_id not in ['0', '99999']:
                d_date_2_dealers[trade_date].append(contra_dealer_id)

        # construct doc list for transactions
        tmp_doc_list = list(map(lambda x: x[1], d_date_2_dealers.items()))

        # construct dictionary for bonds
        tmp_dictionary = corpora.Dictionary(tmp_doc_list)
        num_dealers = len(tmp_dictionary)

        # TODO: check no_below
        # filter bonds whose trade freq is low
        # tmp_dictionary.filter_extremes(no_below=no_below, no_above=1., keep_n=2000)

        # num of dealers after filtering
        no_below_num_dealers = len(tmp_dictionary)

        # calculate the total transaction count and total volume after filtering
        no_below_trace_count = 0
        no_below_volume = 0

        for v in traces:
            volume = v['volume']
            report_dealer_id = v['report_dealer_index']
            contra_dealer_id = v['contra_party_index']

            if report_dealer_id not in ['0', '99999'] and tmp_dictionary.doc2idx([report_dealer_id])[0] != -1:
                no_below_volume += volume
                no_below_trace_count += 1
                continue

            if contra_dealer_id not in ['0', '99999'] and tmp_dictionary.doc2idx([contra_dealer_id])[0] != -1:
                no_below_volume += volume
                no_below_trace_count += 1

        # filter dealers whose total transaction count is low or number of unique bonds is low after filtering
        if no_below_trace_count == 0 or no_below_num_dealers <= 5:
            continue

        # get the unique bonds
        dealer_set = set(list(tmp_dictionary.values()))

        # save features that is needed for clustering
        _l_bonds.append({
            'bond_id': _bond_id,
            'trace_count': tmp_trace_count,
            'volume': tmp_volume,
            'dealer_count': num_dealers,
            'trace_count_after_filtering': no_below_trace_count,
            'volume_after_filtering': no_below_volume,
            'dealer_count_after_filtering': no_below_num_dealers,
            'dealer_set': dealer_set,
            'trace_date_count': len(d_date_2_dealers)
        })

    # sort by total transaction count
    _l_bonds.sort(key=lambda x: -x['trace_count'])

    print(f'skip bonds: {skip_count}')

    # cache data
    utils.write_pkl(cache_path, _l_bonds)
    return _l_bonds


def __similarity(_points, use_bond_overlap=True, use_pattens=False):
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
    #
    # return 0.4 * sim_ret_1 + 0.2 * sim_ret_2 + 0.4 * sim_ret_3


if __name__ == '__main__':
    d_bond_file_name = f'd_bonds_2015_split_by_date.json'

    cluster_num = 5
    group_type = f'k_means_cluster_{cluster_num}_feat_1_trace_count_2_volume_3_num_dealer_split_by_date'
    use_k_means = True
    use_spectral_clustering = False
    use_bond_overlap = False
    use_pattern_info = False
    use_tsne_axis = False

    axis_name = 'x_trace_count_y_distinct_dealer_count_log_scale'
    filtering_for_each_bond = 'no_filtering_axis_first_4400_bonds'
    cache_result_name = f'tmp_group_{group_type}_{axis_name}_{filtering_for_each_bond}.pkl'
    # cache_result_name = 'tmp_graph_dealer_prediction.pkl'

    if use_tsne_axis:
        axis_name = 'tsne'
    else:
        axis_name = 'x_trace_count_y_distinct_dealer_count_log_scale'
    file_name = f'group_{group_type}_{axis_name}_{filtering_for_each_bond}.png'

    cache_result_path = utils.get_relative_file('runtime', 'cache', cache_result_name)
    if 0 and os.path.exists(cache_result_path):
        origin_l_bonds, l_bonds, visual_points, labels = utils.load_pkl(cache_result_path)

    else:
        # generate features for each dealer
        origin_l_bonds = load_l_dealers(use_cache=False)

        print(f'\nnumber of bonds before filtering: {len(origin_l_bonds)}')

        origin_l_bonds = origin_l_bonds[:4400]

        print(f'\nnumber of bonds after filtering: {len(origin_l_bonds)}')

        print('\nConverting ...')

        l_bonds = list(map(lambda x:
                           [
                               x['bond_id'],  # dealer index
                               np.log(x['trace_count']),  # total transaction count without filtering
                               np.log10(x['volume']),  # total volume without filtering
                               x['dealer_count'],  # num of distinct bond without filtering
                               # np.log(x['trace_count_after_filtering']),  # total transaction count after filtering
                               # np.log10(x['volume_after_filtering'] + 1.1),  # total volume after filtering
                               # x['dealer_count_after_filtering'],  # num of distinct dealers after filtering
                               x['dealer_set'],  # bond set
                               # x[14],  # input matrix
                           ],
                           origin_l_bonds))

        print('Normalizing ...')

        # normalize
        points = np.array(list(map(lambda x: x[1:-1], l_bonds)))
        points = (points - np.mean(points, axis=0)) / (np.std(points, axis=0) + 0.001)
        points = points / (np.max(points, axis=0) - np.min(points, axis=0) + 0.001)

        l_bonds = np.array(l_bonds)

        print('Clustering ...')

        # clustering
        if use_spectral_clustering:
            if not use_bond_overlap and not use_pattern_info:
                ret = SpectralClustering(n_clusters=cluster_num).fit(points)
                labels = ret.labels_

            else:
                A = __similarity(l_bonds[:, 1:], use_bond_overlap=use_bond_overlap, use_pattens=use_pattern_info)
                labels = spectral_clustering(A, n_clusters=cluster_num)
        else:
            ret = KMeans(n_clusters=cluster_num).fit(points)
            labels = ret.labels_

        print('Reducing dimensions ...')

        d_bonds_2_group = {}

        labels_path = utils.get_relative_file('runtime', 'dealer_prediction_k_means_4_cluster_labels.json')
        # eco_label_dict = utils.load_json(labels_path)
        # eco_labels = []
        #
        # # assign label for each dealers
        for i, v in enumerate(points):
            bond_id = origin_l_bonds[i]['bond_id']
            d_bonds_2_group[bond_id] = int(labels[i])
        #     print(f'label: {labels[i]}, points: {origin_l_bonds[i][:-2]}')
        #     print(eco_label_dict[str(dealer_index)])
        #     _tmp_label_val = eco_label_dict[str(dealer_index)]
        #     if isinstance(_tmp_label_val, int):
        #         eco_labels.append('peri')
        #     else:
        #         eco_labels.append(_tmp_label_val['core_peri_idb'])

        # save group information to file
        utils.write_json(utils.get_relative_dir('groups_dealer_prediction', f'group_{group_type}.json'),
                         d_bonds_2_group)

        # Dimension reduction so that it can be visualized
        visual_points = ReduceDim.tsne(points, len(points), n_components=2)
        # visual_points = ReduceDim.pca(points, 2)
        # visual_points = np.array(list(map(lambda x: x[1:-2], origin_l_dealers)))[:, [0, 5]]

        utils.write_pkl(utils.get_relative_file('runtime', 'cache',
                                                f'tmp_group_{group_type}_x_trace_count_y_distinct_dealer_count_log_scale_no_filtering_axis.pkl'),
                        [origin_l_bonds, l_bonds, visual_points, labels])

    if use_tsne_axis:
        X, Y = list(zip(*visual_points))
    else:
        X = list(map(lambda a: a['trace_count'], origin_l_bonds))
        Y = list(map(lambda a: a['dealer_count'], origin_l_bonds))

    print('Plotting ...')

    new_labels = list(map(lambda x: f'cluster {x}', labels))

    # new_labels = []
    # for i, v in enumerate(labels):
    #     activity = 'cluster 1'
    #     v = int(v)
    #     # if v == 3:
    #     #     activity = 'low active (cluster 3)'
    #     if v == 2:
    #         activity = 'cluster 2'
    #     elif v == 0:
    #         activity = 'cluster 3'
    #     new_labels.append(activity)

    Visual.spots(X, Y, new_labels, '',
                 # spot_size=3, save_path=utils.get_relative_dir('groups', f'group_{group_type}_tsne_axis.png'),
                 spot_size=3, save_path=utils.get_relative_dir('groups_dealer_prediction', file_name),
                 dict_label_2_color={
                     'cluster 0': 'green',
                     'cluster 1': 'red',
                     'cluster 2': 'blue',
                     'cluster 3': 'black',
                     'cluster 4': 'orange',
                 },
                 x_label='Count of Transactions (Log Scale)' if not use_tsne_axis else 'T-sne axis 1',
                 y_label='Count of Distinct Dealers (Log Scale)' if not use_tsne_axis else 'T-sne axis 2',
                 x_log=False if use_tsne_axis else True,
                 y_log=False if use_tsne_axis else True,
                 legend_size=20)
