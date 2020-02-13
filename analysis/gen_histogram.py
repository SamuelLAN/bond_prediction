import os
import numpy as np
from matplotlib import pyplot as plt
from config import path
from lib import utils

name_map = {
    'buy_client': 'BfC',
    'sell_client': 'StC',
    'buy_dealer': 'BfD',
    'sell_dealer': 'StD',
    'clients': 'BfC+StC',
    'dealers': 'BfD+StD',
    '': 'all',
}

d_dealer = {}

for dir_name in os.listdir(path.PREDICTION_DIR):
    if dir_name[:4] != 'bond':
        continue

    name = dir_name.replace('bonds_by_dealer_', '').replace('bonds_by_dealer', '')
    name = name_map[name]

    print('\n---------------------------------------------')
    print(name)

    transaction_counts = []
    distinct_bond_counts = []
    distinct_contra_dealer_counts = []

    dir_path = os.path.join(path.PREDICTION_DIR, dir_name, '2015')
    for file_name in os.listdir(dir_path):
        if file_name == 'dealer_0.json':
            continue

        file_path = os.path.join(dir_path, file_name)
        data = utils.load_json(file_path)

        # transaction count of each dealer
        counts = len(data)
        transaction_counts.append(counts)

        dealer_index = file_name[len('dealer_'):-len('.json')]
        if dealer_index not in d_dealer:
            d_dealer[dealer_index] = {}
        d_dealer[dealer_index][name] = counts

        # calculate distinct count of bonds and contra_dealer
        d = {}
        d_2 = {}
        for v in data:
            d[v[0]] = True
            d_2[str(v[1])] = True

        distinct_bond_num = len(d)
        distinct_contra_dealer_num = len(d_2)

        distinct_bond_counts.append(distinct_bond_num)
        distinct_contra_dealer_counts.append(distinct_contra_dealer_num)

    print(transaction_counts)
    print(distinct_bond_counts)
    print(distinct_contra_dealer_counts)

    continue

    print('-------')
    print(max(transaction_counts), min(transaction_counts))
    print(max(distinct_bond_counts), min(distinct_bond_counts))
    print(max(distinct_contra_dealer_counts), min(distinct_contra_dealer_counts))
    print(np.histogram(transaction_counts, bins=20))
    print(np.histogram(distinct_bond_counts, bins=20))
    print(np.histogram(distinct_contra_dealer_counts, bins=20))

    plt.figure(1)
    plt.subplot(221)
    bins = list(range(0, 2000, 100))
    plt.hist(transaction_counts, bins=bins)
    plt.title(f"transaction counts of {name} ( range(0, 2000, 100) )")
    plt.subplot(222)
    bins = list(range(2000, 10000, 500))
    plt.hist(transaction_counts, bins=bins)
    plt.title(f"transaction counts of {name} ( range(2000, 10000, 500) )")
    plt.subplot(223)
    bins = list(range(10000, 100000, 10000))
    plt.hist(transaction_counts, bins=bins)
    plt.title(f"transaction counts of {name} ( range(10000, 100000, 10000) )")
    plt.subplot(224)
    bins = list(range(100000, 1000000, 100000))
    plt.hist(transaction_counts, bins=bins)
    plt.title(f"transaction counts of {name} ( range(100000, 1000000, 100000) )")
    plt.show()

    plt.figure(2)
    plt.subplot(211)
    bins = list(range(0, 1000, 100))
    plt.hist(distinct_bond_counts, bins=bins)
    plt.title(f"distinct_bond_counts of {name} ( range(0, 1000, 100) )")
    plt.subplot(212)
    bins = list(range(1000, 10000, 500))
    plt.hist(distinct_bond_counts, bins=bins)
    plt.title(f"distinct_bond_counts of {name} ( range(1000, 10000, 500) )")
    plt.show()

    plt.figure(3)
    plt.subplot(211)
    bins = list(range(0, 100, 5))
    plt.hist(distinct_contra_dealer_counts, bins=bins)
    plt.title(f"distinct_contra_dealer_counts of {name} ( range(0, 100, 5) )")
    plt.subplot(212)
    bins = list(range(100, 500, 10))
    plt.hist(distinct_contra_dealer_counts, bins=bins)
    plt.title(f"distinct_contra_dealer_counts of {name} ( range(100, 500, 10) )")
    plt.show()

print('\n-------------------------------------------')
print(d_dealer)

fractions_of_clients = []
fractions_of_dealers = []

for dealer_index, item in d_dealer.items():
    if 'BfC+StC' in item:
        fractions_of_clients.append(float(item['BfC+StC']) / item['all'])
    if 'BfD+StD' in item:
        fractions_of_dealers.append(float(item['BfD+StD']) / item['all'])

print(fractions_of_clients)
print(fractions_of_dealers)

plt.figure(4)
bins = np.linspace(0., 1., 100)
plt.hist(fractions_of_clients, bins=bins)
plt.title(f"fractions of (BfC+StC)/all ( range(0., 1., 0.01) )")
plt.show()

plt.figure(5)
bins = np.linspace(0., 1., 100)
plt.hist(fractions_of_dealers, bins=bins)
plt.title(f"fractions of (BfD+StD)/all ( range(0., 1., 0.01) )")
plt.show()

