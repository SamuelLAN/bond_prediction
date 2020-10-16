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
total_trade_days = 0

d_dealer_trace_days = {}

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
    format_date = str(date).split(' ')[0]

    if str(offering_date)[0] != '2':
        continue

    offering_timestamp = utils.date_2_timestamp(str(offering_date).split(' ')[0])
    if offering_timestamp >= bound_timestamp:
        continue

    trade_type = ''
    if report_dealer_index == '0':
        trade_type = 'BfC'

        if contra_party_index not in d_dealers:
            d_dealers[contra_party_index] = []
        d_dealers[contra_party_index].append([volume, trade_type, format_date])

        d_dealer_trace_days[(contra_party_index, format_date)] = True

    else:
        if contra_party_index == '99999':
            trade_type = 'StC'

            if report_dealer_index not in d_dealers:
                d_dealers[report_dealer_index] = []
            d_dealers[report_dealer_index].append([volume, trade_type, format_date])
            d_dealer_trace_days[(report_dealer_index, format_date)] = True

        elif contra_party_index != '0':
            trade_type = 'DtD'

            if report_dealer_index not in d_dealers:
                d_dealers[report_dealer_index] = []
            d_dealers[report_dealer_index].append([volume, trade_type, format_date])

            if contra_party_index not in d_dealers:
                d_dealers[contra_party_index] = []
            d_dealers[contra_party_index].append([volume, trade_type, format_date])

            d_dealer_trace_days[(contra_party_index, format_date)] = True

    v['type'] = trade_type

    if bond_id not in d_new_bonds:
        d_new_bonds[bond_id] = []
    d_new_bonds[bond_id].append([volume, trade_type])

    total_volume += volume

total_trade_days = len(d_dealer_trace_days)

print(f'\ntotal_volume: {total_volume}')
print(f'total_transaction_count: {total_transaction_count}')
print(f'num of new bonds: {len(d_new_bonds)}')
print(f'total trace days: {total_trade_days}')

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
    l_dealers.append(
        [dealer_index, len(l), np.sum(list(map(lambda x: x[0], l))), len(set(list(map(lambda x: x[-1], l))))])
l_dealers.sort(key=lambda x: -x[1])

for num_of_dealers in range(260, 460, 20):

    d_first_250_dealers = {}
    for i, v in enumerate(l_dealers[:num_of_dealers]):
        d_first_250_dealers[v[0]] = i

    dealer_total_volume = 0
    dealer_total_transaction_count = 0
    d_dealer_new_bond = {}
    d_date = {}

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
        format_date = str(date).split(' ')[0]

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

            if report_dealer_index in d_first_250_dealers:
                d_date[(report_dealer_index, format_date)] = True
            elif contra_party_index in d_first_250_dealers:
                d_date[(contra_party_index, format_date)] = True

    print(
        f'\ntotal transaction count of dealers within first {num_of_dealers} dealers: {dealer_total_transaction_count} ({dealer_total_transaction_count / total_transaction_count * 100.}%)')
    print(
        f'total volume of dealers within first {num_of_dealers} dealers: {dealer_total_volume} ({dealer_total_volume / total_volume * 100.}%)')
    print(
        f'num of new bonds of dealers within first {num_of_dealers} dealers: {len(d_dealer_new_bond)} ({len(d_dealer_new_bond) / len(d_new_bonds) * 100.}%)')
    print(
        f'total trace days within first {num_of_dealers} dealers: {len(d_date)} ({len(d_date) / total_trade_days * 100.}%)')

# total_volume: 2670541660818.1377
# total_transaction_count: 6735448
# num of new bonds: 8602
# total trace days: 92893
#  
# total transaction count of dealers within first 20 dealers: 3597283 (53.40822169512703%)
# total volume of dealers within first 20 dealers: 1731954731349.905 (64.85406150972793%)
# num of new bonds of dealers within first 20 dealers: 8315 (96.66356661241572%)
# total trace days within first 20 dealers: 5021 (5.405143552259051%)
#  
# total transaction count of dealers within first 40 dealers: 4304846 (63.91328386768037%)
# total volume of dealers within first 40 dealers: 2122395636893.4922 (79.47435039239502%)
# num of new bonds of dealers within first 40 dealers: 8414 (97.81446175308068%)
# total trace days within first 40 dealers: 10032 (10.799522030723521%)
#  
# total transaction count of dealers within first 60 dealers: 4604613 (68.3638712673604%)
# total volume of dealers within first 60 dealers: 2323608786571.164 (87.00889488686394%)
# num of new bonds of dealers within first 60 dealers: 8468 (98.44222273889794%)
# total trace days within first 60 dealers: 15032 (16.18205892801395%)
#  
# total transaction count of dealers within first 80 dealers: 4755455 (70.60339564643658%)
# total volume of dealers within first 80 dealers: 2455789943792.377 (91.95849590453608%)
# num of new bonds of dealers within first 80 dealers: 8490 (98.69797721460125%)
# total trace days within first 80 dealers: 20029 (21.561366303166007%)
#  
# total transaction count of dealers within first 100 dealers: 4855332 (72.08625172371607%)
# total volume of dealers within first 100 dealers: 2542763975743.9375 (95.2152895815505%)
# num of new bonds of dealers within first 100 dealers: 8565 (99.56986747268077%)
# total trace days within first 100 dealers: 24680 (26.568202125025568%)
#  
# total transaction count of dealers within first 120 dealers: 4920002 (73.04639572601556%)
# total volume of dealers within first 120 dealers: 2565515501488.117 (96.06723381735804%)
# num of new bonds of dealers within first 120 dealers: 8569 (99.61636828644501%)
# total trace days within first 120 dealers: 29238 (31.474922760595526%)
#  
# total transaction count of dealers within first 140 dealers: 4967186 (73.74692819245283%)
# total volume of dealers within first 140 dealers: 2594122802959.2573 (97.13845101238866%)
# num of new bonds of dealers within first 140 dealers: 8578 (99.72099511741456%)
# total trace days within first 140 dealers: 34014 (36.616322004887344%)
#  
# total transaction count of dealers within first 160 dealers: 4999478 (74.22636178024091%)
# total volume of dealers within first 160 dealers: 2608685275969.878 (97.68375136191248%)
# num of new bonds of dealers within first 160 dealers: 8579 (99.73262032085562%)
# total trace days within first 160 dealers: 38540 (41.48859440431465%)
#  
# total transaction count of dealers within first 180 dealers: 5021678 (74.55596123672844%)
# total volume of dealers within first 180 dealers: 2616245796707.878 (97.96685949869713%)
# num of new bonds of dealers within first 180 dealers: 8581 (99.75587072773774%)
# total trace days within first 180 dealers: 42662 (45.92595782244087%)
#  
# total transaction count of dealers within first 200 dealers: 5046007 (74.91716957802956%)
# total volume of dealers within first 200 dealers: 2629841181971.9976 (98.47594667990795%)
# num of new bonds of dealers within first 200 dealers: 8583 (99.77912113461986%)
# total trace days within first 200 dealers: 47058 (50.65828426253862%)
#  
# total transaction count of dealers within first 220 dealers: 5066607 (75.2230141187342%)
# total volume of dealers within first 220 dealers: 2635570605956.008 (98.69048832395237%)
# num of new bonds of dealers within first 220 dealers: 8586 (99.81399674494304%)
# total trace days within first 220 dealers: 51324 (55.25066474330681%)
#  
# total transaction count of dealers within first 240 dealers: 5081914 (75.45027442866457%)
# total volume of dealers within first 240 dealers: 2640802932341.787 (98.88641585665285%)
# num of new bonds of dealers within first 240 dealers: 8586 (99.81399674494304%)
# total trace days within first 240 dealers: 55011 (59.219747451368775%)
#  
# total transaction count of dealers within first 260 dealers: 5094865 (75.64255562510466%)
# total volume of dealers within first 260 dealers: 2643211603214.207 (98.97660995127266%)
# num of new bonds of dealers within first 260 dealers: 8587 (99.8256219483841%)
# total trace days within first 260 dealers: 58439 (62.9100147481511%)
# 
#  
# total transaction count of dealers within first 280 dealers: 5104649 (75.78781693511701%)
# total volume of dealers within first 280 dealers: 2644125840873.127 (99.0108441170351%)
# num of new bonds of dealers within first 280 dealers: 8587 (99.8256219483841%)
# total trace days within first 280 dealers: 61181 (65.86179798262518%)
#  
# total transaction count of dealers within first 300 dealers: 5112964 (75.9112682630762%)
# total volume of dealers within first 300 dealers: 2646339864273.127 (99.09374952279919%)
# num of new bonds of dealers within first 300 dealers: 8587 (99.8256219483841%)
# total trace days within first 300 dealers: 64252 (69.16775214494095%)
#  
# total transaction count of dealers within first 320 dealers: 5120931 (76.02955289685259%)
# total volume of dealers within first 320 dealers: 2647939528448.127 (99.15364988677666%)
# num of new bonds of dealers within first 320 dealers: 8587 (99.8256219483841%)
# total trace days within first 320 dealers: 66989 (72.11415284251773%)
#  
# total transaction count of dealers within first 340 dealers: 5127079 (76.12083116074832%)
# total volume of dealers within first 340 dealers: 2650804581093.127 (99.26093346475021%)
# num of new bonds of dealers within first 340 dealers: 8588 (99.83724715182515%)
# total trace days within first 340 dealers: 69588 (74.9119955217293%)
#  
# total transaction count of dealers within first 360 dealers: 5132260 (76.19775254741779%)
# total volume of dealers within first 360 dealers: 2653648882093.127 (99.36743998519628%)
# num of new bonds of dealers within first 360 dealers: 8588 (99.83724715182515%)
# total trace days within first 360 dealers: 71957 (77.46224150366551%)
#  
# total transaction count of dealers within first 380 dealers: 5137790 (76.27985547509238%)
# total volume of dealers within first 380 dealers: 2656655214235.107 (99.48001385686015%)
# num of new bonds of dealers within first 380 dealers: 8588 (99.83724715182515%)
# total trace days within first 380 dealers: 74108 (79.77780887687985%)
#  
# total transaction count of dealers within first 400 dealers: 5142226 (76.34571597910042%)
# total volume of dealers within first 400 dealers: 2657867261069.107 (99.52539966198663%)
# num of new bonds of dealers within first 400 dealers: 8588 (99.83724715182515%)
# total trace days within first 400 dealers: 75918 (81.726287233699%)
#  
# total transaction count of dealers within first 420 dealers: 5145521 (76.39463625879081%)
# total volume of dealers within first 420 dealers: 2659000242067.1978 (99.5678248004787%)
# num of new bonds of dealers within first 420 dealers: 8588 (99.83724715182515%)
# total trace days within first 420 dealers: 77753 (83.70167827500458%)
#  
# total transaction count of dealers within first 440 dealers: 5147851 (76.42922935489963%)
# total volume of dealers within first 440 dealers: 2659578233532.9478 (99.58946803017366%)
# num of new bonds of dealers within first 440 dealers: 8588 (99.83724715182515%)
# total trace days within first 440 dealers: 79021 (85.06668963215743%)
