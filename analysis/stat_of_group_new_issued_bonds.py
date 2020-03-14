import os
import numpy as np
from gensim import corpora
from matplotlib import pyplot as plt
from config import path
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
# d = {}
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
#     if report_dealer_index == '0':
#         trade_type = 'BfC'
#
#         if contra_party_index not in d_dealers:
#             d_dealers[contra_party_index] = []
#         d_dealers[contra_party_index].append([bond_id, volume, trade_type, date_clean])
#
#     else:
#         if contra_party_index == '99999':
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
#             if contra_party_index not in d_dealers:
#                 d_dealers[contra_party_index] = []
#             d_dealers[contra_party_index].append([bond_id, volume, trade_type, date_clean])
#
#     v['type'] = trade_type
#
#     if trade_type not in d:
#         d[trade_type] = 0
#     d[trade_type] += 1
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
# # # l_bonds = []
# # # for bond_id, l in d_new_bonds.items():
# # #     l_bonds.append([bond_id, len(l), np.sum(list(map(lambda x: x[0], l)))])
# # # l_bonds.sort(key=lambda x: -x[1])
# #
# if '0' in d_dealers:
#     del d_dealers['0']
# if '99999' in d_dealers:
#     del d_dealers['99999']
#
# print(d)

from six.moves import cPickle as pickle

# with open(os.path.join(path.ROOT_DIR, 'runtime', 'tmp123.pkl'), 'wb') as f:
#     pickle.dump([data, d_dealers, total_volume, total_transaction_count, bound_timestamp, d_new_bonds], f)

with open(os.path.join(path.ROOT_DIR, 'runtime', 'tmp123.pkl'), 'rb') as f:
    data, d_dealers, total_volume, total_transaction_count, bound_timestamp, d_new_bonds = pickle.load(f)

d_dealer_for_gen_input = {}

l_dealers = []
for dealer_index, l in d_dealers.items():
    tmp_trace_count = len(l)
    if tmp_trace_count < 1000:
        continue

    tmp_volume = np.sum(list(map(lambda x: x[1], l)))
    new_l = [dealer_index, tmp_trace_count, tmp_volume]

    if tmp_trace_count > 100000:
        no_below_list = [40, 50]
    elif tmp_trace_count > 10000:
        no_below_list = [20, 25]
    else:
        no_below_list = [5, 10]

    tmp_d_bonds = {}
    for v in l:
        bond_id = v[0]
        offering_date = v[-1]
        if offering_date not in tmp_d_bonds:
            tmp_d_bonds[offering_date] = []
        tmp_d_bonds[offering_date].append(bond_id)

    tmp_doc_list = list(map(lambda x: x[1], tmp_d_bonds.items()))
    tmp_bonds = list(map(lambda x: x[0], l))

    for no_below in no_below_list:
        tmp_dictionary = corpora.Dictionary(tmp_doc_list)
        tmp_dictionary.filter_extremes(no_below=no_below, no_above=1., keep_n=2000)
        no_below_num_bonds = len(tmp_dictionary)

        tmp_bond_indices = tmp_dictionary.doc2idx(tmp_bonds)
        no_below_trace_count = 0
        no_below_volume = 0

        for i, v in enumerate(tmp_bond_indices):
            if v == -1:
                continue
            no_below_trace_count += 1
            no_below_volume += l[i][1]

        new_l.append([no_below_trace_count, no_below_volume, no_below_num_bonds, tmp_dictionary])

    if dealer_index not in d_dealer_for_gen_input:
        tmp_dictionary = new_l[-1][-1]

        tmp_trace_list = []
        for i, v in enumerate(l):
            tmp_bond_id = v[0]
            if tmp_dictionary.doc2idx([tmp_bond_id])[0] == -1:
                continue
            tmp_trace_list.append(v)

        d_dealer_for_gen_input[dealer_index] = {
            'dealer_index': dealer_index,
            'total_transaction_count': tmp_trace_count,
            'total_volume': tmp_volume,
            'dictionary': tmp_dictionary,
            'no_below_transaction_count': new_l[-1][0],
            'no_below_volume': new_l[-1][1],
            'no_below_num_bonds': new_l[-1][2],
            'trace_list': tmp_trace_list,
        }

    if new_l[-1][0] == 0 or new_l[-1][2] <= 5:
        continue
    l_dealers.append(new_l)
l_dealers.sort(key=lambda x: -x[1])

print(f'len of d_dealer_for_gen_input: {len(d_dealer_for_gen_input)}')
utils.write_pkl(os.path.join(path.ROOT_DIR, 'runtime', 'd_dealer_for_gen_input_with_no_below_50_25_10.pkl'), d_dealer_for_gen_input)
# print('done')
# exit()


string = 'num_of_dealers,dealers_total_transaction_count,dealers_total_transaction_count(percentage),'
string += 'dealer_total_volume,dealer_total_volume(percentage),'
string += 'num_of_old_bonds,num_of_old_bonds(percentage),'
string += 'dealers_total_transaction_count_2,dealers_total_transaction_count_2(percentage),'
string += 'dealer_total_volume_2,dealer_total_volume_2(percentage),'
string += 'num_of_old_bonds_2,num_of_old_bonds_2(percentage)\n'

for num_of_dealers in range(20, 270, 20):

    tmp_dealers = l_dealers[:num_of_dealers]
    dict_first_num_dealers = {}
    for v in tmp_dealers:
        dict_first_num_dealers[v[0]] = True

    dealer_total_volume = 0
    dealer_total_transaction_count = 0
    d_dealer_new_bond = {}

    dealer_total_volume_2 = 0
    dealer_total_transaction_count_2 = 0
    d_dealer_new_bond_2 = {}

    length = len(data)
    for i, v in enumerate(data):
        if i % 20 == 0:
            progress = float(i) / length * 100.
            print('\rprogress: %.2f%% ' % progress, end='')

        bond_id = v['bond_id']
        offering_date = v['offering_date']
        report_dealer_index = str(v['report_dealer_index'])
        contra_party_index = str(v['contra_party_index'])
        date = v['date']
        volume = v['volume']

        if str(offering_date)[0] != '2':
            continue

        offering_timestamp = utils.date_2_timestamp(str(offering_date).split(' ')[0])
        if offering_timestamp >= bound_timestamp:
            continue

        if report_dealer_index not in dict_first_num_dealers and contra_party_index not in dict_first_num_dealers:
            continue

        in_bond = False
        for _dealer in tmp_dealers:
            no_below_0_val = _dealer[3]
            no_below_dict_0 = no_below_0_val[-1]
            if no_below_dict_0.doc2idx([bond_id])[0] != -1:
                in_bond = True
                break

        if in_bond:
            dealer_total_transaction_count += 1
            dealer_total_volume += volume

            if bond_id not in d_dealer_new_bond:
                d_dealer_new_bond[bond_id] = True

        in_bond = False
        for _dealer in tmp_dealers:
            no_below_1_val = _dealer[4]
            no_below_dict_1 = no_below_1_val[-1]
            if no_below_dict_1.doc2idx([bond_id])[0] != -1:
                in_bond = True
                break

        if in_bond:
            dealer_total_transaction_count_2 += 1
            dealer_total_volume_2 += volume

            if bond_id not in d_dealer_new_bond_2:
                d_dealer_new_bond_2[bond_id] = True

    string += f'{num_of_dealers},'
    string += f'{dealer_total_transaction_count},{dealer_total_transaction_count / total_transaction_count * 100.},'
    string += f'{dealer_total_volume},{dealer_total_volume / total_volume * 100.},'
    string += f'{len(d_dealer_new_bond)},{len(d_dealer_new_bond) / len(d_new_bonds) * 100.},'

    string += f'{dealer_total_transaction_count_2},{dealer_total_transaction_count_2 / total_transaction_count * 100.},'
    string += f'{dealer_total_volume_2},{dealer_total_volume_2 / total_volume * 100.},'
    string += f'{len(d_dealer_new_bond_2)},{len(d_dealer_new_bond_2) / len(d_new_bonds) * 100.}\n'

    print(
        f'\ntotal transaction count of dealers within first {num_of_dealers} transaction count: {dealer_total_transaction_count} ({dealer_total_transaction_count / total_transaction_count * 100.}%)')
    print(
        f'total volume of dealers within first {num_of_dealers} transaction count: {dealer_total_volume} ({dealer_total_volume / total_volume * 100.}%)')
    print(
        f'num of new bonds of dealers within first {num_of_dealers} transaction count: {len(d_dealer_new_bond)} ({len(d_dealer_new_bond) / len(d_new_bonds) * 100.}%)')

    print(
        f'\n2. total transaction count of dealers within first {num_of_dealers} transaction count: {dealer_total_transaction_count_2} ({dealer_total_transaction_count_2 / total_transaction_count * 100.}%)')
    print(
        f'2. total volume of dealers within first {num_of_dealers} transaction count: {dealer_total_volume_2} ({dealer_total_volume_2 / total_volume * 100.}%)')
    print(
        f'2. num of new bonds of dealers within first {num_of_dealers} transaction count: {len(d_dealer_new_bond_2)} ({len(d_dealer_new_bond_2) / len(d_new_bonds) * 100.}%)')

with open(os.path.join(path.ROOT_DIR, 'runtime', 'stat_after_no_below_40_20_5_50_25_10.csv'), 'w') as f:
    f.write(string)
