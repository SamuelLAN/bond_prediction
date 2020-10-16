import os
import numpy as np
from config import path
from lib import utils
from gensim import corpora
import matplotlib.pyplot as plt


def load_d_dealers_after_filtering(_trace_suffix, use_cache=True):
    print('\nloading d_dealers_2_traces ... ')

    cache_path = utils.get_relative_dir('runtime', 'cache', 'filtered_d_dealers.pkl')
    if os.path.exists(cache_path) and use_cache:
        return utils.load_pkl(cache_path)

    # load trace data
    train_d_dealers = utils.load_json(os.path.join(path.D_DEALERS_TRACE_DIR, f'train_{_trace_suffix}'))
    test_d_dealers = utils.load_json(os.path.join(path.D_DEALERS_TRACE_DIR, f'test_{_trace_suffix}'))

    # remove clients, we do not predict clients' behaviors
    if '0' in train_d_dealers:
        del train_d_dealers['0']
    if '99999' in train_d_dealers:
        del train_d_dealers['99999']
    if '0' in test_d_dealers:
        del test_d_dealers['0']
    if '99999' in test_d_dealers:
        del test_d_dealers['99999']

    new_train_d_dealers = {}
    new_test_d_dealers = {}

    length = len(train_d_dealers)
    count = 0

    for _dealer_index, train_traces in train_d_dealers.items():
        # output progress
        count += 1
        if count % 2 == 0:
            progress = float(count) / length * 100.
            print('\rprogress: %.2f%% ' % progress, end='')

        test_traces = test_d_dealers[_dealer_index]

        # calculate the total transaction count
        train_trace_count = len(train_traces)

        # filter bonds whose trade frequency is low
        if train_trace_count < 1000:
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


cache_path = utils.get_relative_dir('runtime', 'cache', 'tmp_for_states_l_dealers.pkl')
if os.path.exists(cache_path):
    l_dealers = utils.load_pkl(cache_path)
else:
    # load group_dict
    group_name = 'group_k_means_split_by_date.json'
    group_file_path = os.path.join(path.ROOT_DIR, 'groups', group_name)
    d_dealer_index_2_group_label = utils.load_json(group_file_path)

    # load d dealers and filter traces and bonds whose freq is low
    trace_suffix = 'd_dealers_2015_split_by_date.json'
    train_d_dealers, test_d_dealers = load_d_dealers_after_filtering(trace_suffix, use_cache=True)

    d_dealer_bonds = {}
    for dealer_index, trace_list in train_d_dealers.items():
        bond_set = list(set(list(map(lambda x: x[0], trace_list))))
        d_dealer_bonds[dealer_index] = bond_set

    # get total train trace list
    train_trace_list = []
    for dealer_index, traces in train_d_dealers.items():
        if dealer_index not in d_dealer_index_2_group_label:
            continue
        train_trace_list += traces

        # print(dealer_index, len(trace_list), trace_list)

    # sort trace list so that bond traded first could be get smaller bond index (for visualization convenience)
    train_trace_list.sort(key=lambda x: x[-1])

    # get dictionary
    bond_list = list(map(lambda x: x[0], train_trace_list))
    train_doc_list = [bond_list]
    dictionary = corpora.Dictionary(train_doc_list)
    len_bonds = len(dictionary)
    bond_list = list(set(bond_list))

    print(f'total bond num after filtering: {len_bonds}\n')

    cache_name = utils.get_relative_dir('runtime', 'cache', 'stats.pkl')
    if os.path.exists(cache_name):
        data = utils.load_pkl(cache_name)

    else:
        pkl_path = os.path.join(path.TRACE_DIR, 'finra_trace_2015.pkl')
        data = utils.load_pkl(pkl_path)

        columns = data.columns
        for i, v in enumerate(columns):
            print(i, v)

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

        utils.write_pkl(cache_name, data)

    # initialize some useful variables
    total_volume = 0.
    total_transaction_count = len(data)
    total_bond_num = 0
    total_bond_set = set()
    d_dealers = {}

    data_dealers = {}
    total_transaction_count_before_filtering = 0
    total_volume_before_filtering = 0
    bond_set_before_filtering = set()

    total_transaction_count_after_filtering = 0
    total_volume_after_filtering = 0
    bond_set_after_filtering = set()

    total_transaction_count_after_filtering_train = 0
    total_volume_after_filtering_train = 0

    total_transaction_count_after_filtering_test = 0
    total_volume_after_filtering_test = 0

    bound_train = '2015-11-25'

    type_list = []
    filter_new_bond_date = '2014-06-01'

    # traverse data
    length = len(data)
    for i, v in enumerate(data):
        # show progress
        if i % 20 == 0:
            progress = float(i) / length * 100.
            print('\rprogress: %.2f%% ' % progress, end='')

        # get data
        bond_id = v['bond_id']
        offering_date = str(v['offering_date'])
        report_dealer_index = v['report_dealer_index']
        contra_party_index = v['contra_party_index']
        date = str(v['date'])
        volume = v['volume']

        total_volume += volume
        total_bond_set.add(bond_id)

        trade_type = ''
        if report_dealer_index == 0:
            trade_type = 'bfc'

            if contra_party_index not in d_dealers:
                d_dealers[contra_party_index] = []
            d_dealers[contra_party_index].append([bond_id, volume, trade_type, offering_date, date])

        else:
            if contra_party_index == 99999:
                trade_type = 'stc'

                if report_dealer_index not in d_dealers:
                    d_dealers[report_dealer_index] = []
                d_dealers[report_dealer_index].append([bond_id, volume, trade_type, offering_date, date])

            elif contra_party_index != 0:
                trade_type = 'dtd'

                if report_dealer_index not in d_dealers:
                    d_dealers[report_dealer_index] = []
                d_dealers[report_dealer_index].append([bond_id, volume, trade_type, offering_date, date])

                if contra_party_index not in d_dealers:
                    d_dealers[contra_party_index] = []
                d_dealers[contra_party_index].append([bond_id, volume, trade_type, offering_date, date])

        if not trade_type:
            continue

        type_list.append(trade_type)

        # filtering

        # filter all transactions from new issued bonds
        if offering_date > filter_new_bond_date or offering_date[0] != '2':
            continue

        report_dealer_index = str(report_dealer_index)
        contra_party_index = str(contra_party_index)

        if (report_dealer_index in train_d_dealers or contra_party_index in train_d_dealers) and \
                (report_dealer_index in d_dealer_index_2_group_label or contra_party_index in d_dealer_index_2_group_label):
            pass
        else:
            continue

        if report_dealer_index in train_d_dealers and report_dealer_index in d_dealer_index_2_group_label:
            data_dealers[report_dealer_index] = 1
        if contra_party_index in train_d_dealers and contra_party_index in d_dealer_index_2_group_label:
            data_dealers[contra_party_index] = 1

        total_transaction_count_before_filtering += 1
        total_volume_before_filtering += volume
        bond_set_before_filtering.add(bond_id)

        tmp_bond_list = d_dealer_bonds[report_dealer_index] if report_dealer_index in d_dealer_bonds else \
            d_dealer_bonds[contra_party_index]
        if bond_id not in tmp_bond_list:
            if contra_party_index not in d_dealer_bonds:
                continue
            tmp_bond_list = d_dealer_bonds[contra_party_index]
            if bond_id not in tmp_bond_list:
                continue

        total_transaction_count_after_filtering += 1
        total_volume_after_filtering += volume
        bond_set_after_filtering.add(bond_id)

        if date >= bound_train:
            total_transaction_count_after_filtering_test += 1
            total_volume_after_filtering_test += volume
        else:
            total_transaction_count_after_filtering_train += 1
            total_volume_after_filtering_train += volume

    # summary the daata
    total_bond_num = len(total_bond_set)
    type_list = np.array(type_list)

    if '0' in data_dealers:
        del data_dealers['0']
    if '99999' in data_dealers:
        del data_dealers['99999']

    # show data stats
    print('\n\n-------------------------------------')
    print(f'total_transaction_count: {total_transaction_count}')
    print(f'total_volume: {total_volume}')
    print(f'total_bond_num: {total_bond_num}')
    print(f'total_dealer_num: {len(d_dealers)}')

    print(f'total_transaction_count before filtering: {total_transaction_count_before_filtering}')
    print(f'total_volume before filtering: {total_volume_before_filtering}')
    print(f'total_bond_num before filtering: {len(bond_set_before_filtering)}')
    print(f'total_dealer_num before filtering: {len(data_dealers)}')

    print(f'total_transaction_count after filtering: {total_transaction_count_after_filtering}')
    print(f'total_volume after filtering: {total_volume_after_filtering}')
    print(f'total_bond_num after filtering: {len(bond_set_after_filtering)}')
    print(f'total_dealer_num after filtering: {len(data_dealers)}')

    print(f'total_transaction_count after filtering train: {total_transaction_count_after_filtering_train}')
    print(f'total_volume after filtering train: {total_volume_after_filtering_train}')

    print(f'total_transaction_count after filtering test: {total_transaction_count_after_filtering_test}')
    print(f'total_volume after filtering test: {total_volume_after_filtering_test}')

    print(f'num of BfC: {len(type_list[np.argwhere(type_list == "bfc")])}')
    print(f'num of SfC: {len(type_list[np.argwhere(type_list == "stc")])}')
    print(f'num of DtD: {len(type_list[np.argwhere(type_list == "dtd")])}')

    l_dealers = list(d_dealers.items())

    utils.write_pkl(cache_path, l_dealers)


# for hist
transaction_count_dealers = list(map(lambda x: len(x[1]), l_dealers))
# transaction_count_dealers = list(filter(lambda x: x <= 16000, transaction_count_dealers))
# hist, bins = np.histogram(transaction_count_dealers, 30)

# distinct_bond_count_dealers = list(map(lambda x: len(set(list(map(lambda a: a[0], x[1])))), l_dealers))

# bins = list(range(0, 11000, 1000)) + list(range(10000, 18000, 2000))
bins = [0, 10000, 25000, 50000, 75000, 100000, 1000000]
# bins = list(range(0, 11000, 1000)) + list(range(10000, 18000, 2000))
# bins = 40

plt.figure(figsize=(18., 18 * 4.8 / 10.4))
plt.hist(transaction_count_dealers, label='transaction count', bins=bins, color='#6060FF', edgecolor='#E6E6E6')
# plt.hist(distinct_bond_count_dealers, label='distinct bond count', bins=bins, color='#E68080', edgecolor='#E6E6E6')
# plt.title('Histogram of transaction count per dealer', fontsize=30)
plt.xlabel('Count of transactions', fontsize=28)
plt.ylabel('Count of dealers', fontsize=28)
# plt.xticks(list(map(int, range(0, 33000, 3000))))
# plt.xticks(list(range(0, 12000, 2000)) + list(range(10000, 18000, 2000)), fontsize=20)
plt.xticks(bins, list(map(lambda a: str(a)[:2], bins)), fontsize=20)
plt.yticks(list(map(int, range(0, 1100, 100))), fontsize=20)
plt.grid(linestyle='dashed', axis='y')

plt.yscale('log')
plt.xscale('log')

# plt.legend()
plt.savefig(utils.get_relative_dir('runtime', 'hist', 'hist_of_transaction_count_per_dealer11.png'), dpi=500)
plt.show()


# total_transaction_count: 6735448
# total_volume: 3964704770060.5776
# total_bond_num: 12352
# total_dealer_num: 1249
# num of BfC: 1592996
# num of SfC: 2275174
# num of DtD: 2841166


# print(f'train set dealers: {len(train_d_dealers)}')
# print(f'test set dealers: {len(test_d_dealers)}')

# -------------------------------------
# total_transaction_count: 6735448
# total_volume: 3964704770060.5776
# total_bond_num: 12352
# total_dealer_num: 1249

# total_transaction_count before filtering: 6513713
# total_volume before filtering: 3855236427660.722
# total_bond_num before filtering: 12324
# total_dealer_num before filtering: 199

# filter new issued bonds
# total_transaction_count before filtering: 5000311
# total_volume before filtering: 2592392955895.688
# total_bond_num before filtering: 8580
# total_dealer_num before filtering: 199

# total_transaction_count after filtering: 3165542
# total_volume after filtering: 1244214557389.0496
# total_bond_num after filtering: 4438
# total_dealer_num after filtering: 199

# total_transaction_count after filtering train: 2906823
# total_volume after filtering train: 1162079612168.1196
# total_transaction_count after filtering test: 258719
# total_volume after filtering test: 82134945220.93004

# num of BfC: 1592996
# num of SfC: 2275174
# num of DtD: 2841166
# train set dealers: 206
# test set dealers: 206
