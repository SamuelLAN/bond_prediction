import os
import numpy as np
from matplotlib import pyplot as plt
from config import path
from lib import utils

path_pkl_2015 = os.path.join(path.TRACE_DIR, 'finra_trace_2015.pkl')
data = utils.load_pkl(path_pkl_2015)

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

d_new_bonds = {}
d_dealers = {}

total_volume = 0.
total_transaction_count = len(data)

bound_timestamp = utils.date_2_timestamp('2014-06-01')

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

    trade_type = ''
    if report_dealer_index == 0:
        trade_type = 'BfC'

        if contra_party_index not in d_dealers:
            d_dealers[contra_party_index] = []
        d_dealers[contra_party_index].append([volume, trade_type])

    else:
        if contra_party_index == 99999:
            trade_type = 'StC'

            if report_dealer_index not in d_dealers:
                d_dealers[report_dealer_index] = []
            d_dealers[report_dealer_index].append([volume, trade_type])

        else:
            trade_type = 'DtD'

            if report_dealer_index not in d_dealers:
                d_dealers[report_dealer_index] = []
            d_dealers[report_dealer_index].append([volume, trade_type])

            if contra_party_index not in d_dealers:
                d_dealers[contra_party_index] = []
            d_dealers[contra_party_index].append([volume, trade_type])

    v['type'] = trade_type

    if bond_id not in d_new_bonds:
        d_new_bonds[bond_id] = []
    d_new_bonds[bond_id].append([volume, trade_type])

    total_volume += volume

print(f'\ntotal_volume: {total_volume}')
print(f'total_transaction_count: {total_transaction_count}')
print(f'num of new bonds: {len(d_new_bonds)}')

# l_bonds = []
# for bond_id, l in d_new_bonds.items():
#     l_bonds.append([bond_id, len(l), np.sum(list(map(lambda x: x[0], l)))])
# l_bonds.sort(key=lambda x: -x[1])

if '0' in d_dealers:
    del d_dealers['0']
if '99999' in d_dealers:
    del d_dealers['99999']

l_dealers = []
for dealer_index, l in d_dealers.items():
    l_dealers.append([dealer_index, len(l), np.sum(list(map(lambda x: x[0], l)))])
l_dealers.sort(key=lambda x: -x[1])

for num_of_dealers in range(20, 270, 20):

    d_first_250_dealers = {}
    for i, v in enumerate(l_dealers[:num_of_dealers]):
        d_first_250_dealers[v[0]] = i

    dealer_total_volume = 0
    dealer_total_transaction_count = 0
    d_dealer_new_bond = {}

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

        if report_dealer_index in d_first_250_dealers or contra_party_index in d_first_250_dealers:
            dealer_total_transaction_count += 1
            dealer_total_volume += volume

            if bond_id not in d_dealer_new_bond:
                d_dealer_new_bond[bond_id] = True

    print(
        f'\ntotal transaction count of dealers within first {num_of_dealers} transaction count: {dealer_total_transaction_count} ({dealer_total_transaction_count / total_transaction_count * 100.}%)')
    print(
        f'total volume of dealers within first {num_of_dealers} transaction count: {dealer_total_volume} ({dealer_total_volume / total_volume * 100.}%)')
    print(
        f'num of new bonds of dealers within first {num_of_dealers} transaction count: {len(d_dealer_new_bond)} ({len(d_dealer_new_bond) / len(d_new_bonds) * 100.}%)')
