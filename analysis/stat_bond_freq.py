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
    _volume = v['volume']

    if str(offering_date)[0] != '2':
        continue

    offering_timestamp = utils.date_2_timestamp(str(offering_date).split(' ')[0])
    if offering_timestamp >= bound_timestamp:
        continue

    if bond_id not in d_new_bonds:
        d_new_bonds[bond_id] = {'count': 0, 'volume': 0}
    d_new_bonds[bond_id]['count'] += 1
    d_new_bonds[bond_id]['volume'] += _volume

l_bonds_count = list(map(lambda x: [x[0], x[1]['count']], d_new_bonds.items()))
l_bonds_volume = list(map(lambda x: [x[0], x[1]['volume']], d_new_bonds.items()))

l_bonds_count.sort(key=lambda x: x[1])
l_bonds_volume.sort(key=lambda x: x[1])

mean_count = np.mean(list(map(lambda x: x[1], l_bonds_count)))
mean_volume = np.mean(list(map(lambda x: x[1], l_bonds_volume)))

std_count = np.std(list(map(lambda x: x[1], l_bonds_count)))
std_volume = np.std(list(map(lambda x: x[1], l_bonds_volume)))

d_bond_id_2_freq_type_by_count = {}
d_bond_id_2_freq_type_by_volume = {}

print(f'mean_count: {mean_count}, std_count: {std_count}')
print(f'mean_volume: {mean_volume}, std_volume: {std_volume}')

for bond_id, count in l_bonds_count:
    if count >= mean_count + std_count:
        d_bond_id_2_freq_type_by_count[bond_id] = 'most'
    elif mean_count + std_count > count >= mean_count:
        d_bond_id_2_freq_type_by_count[bond_id] = 'more'
    elif mean_count > count >= mean_count - std_count * 0.3:
        d_bond_id_2_freq_type_by_count[bond_id] = 'less'
    else:
        d_bond_id_2_freq_type_by_count[bond_id] = 'least'

for bond_id, _volume in l_bonds_volume:
    if _volume >= mean_volume + std_volume:
        d_bond_id_2_freq_type_by_volume[bond_id] = 'most'
    elif mean_volume + std_volume > _volume >= mean_volume:
        d_bond_id_2_freq_type_by_volume[bond_id] = 'more'
    elif mean_volume > _volume >= mean_volume - std_volume * 0.3:
        d_bond_id_2_freq_type_by_volume[bond_id] = 'less'
    else:
        d_bond_id_2_freq_type_by_volume[bond_id] = 'least'

path_d_bond_id_2_freq_type_by_count = utils.get_relative_dir('runtime', 'cache', 'd_bond_id_2_freq_type_by_count.json')
utils.write_json(path_d_bond_id_2_freq_type_by_count, d_bond_id_2_freq_type_by_count)

path_d_bond_id_2_freq_type_by_volume = utils.get_relative_dir(
    'runtime', 'cache', 'd_bond_id_2_freq_type_by_volume.json')
utils.write_json(path_d_bond_id_2_freq_type_by_volume, d_bond_id_2_freq_type_by_volume)

print('\ndone')

# progress: 100.00% mean_count: 601.26331085794, std_count: 1151.961991547412
# mean_volume: 310455901.0483783, std_volume: 514301578.29584545
