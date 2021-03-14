from lib import data
import numpy as np

d_dealer_2_dealer = data.load_dealer_2_dealer()

interact_dealer = list(map(lambda x: len(x[1]), d_dealer_2_dealer.items()))
interact_mean_count = list(map(lambda x: np.mean(
    list(map(lambda a: a[1]['count'], x[1].items()))), d_dealer_2_dealer.items()))
interact_mean_volume = list(map(lambda x: np.mean(
    list(map(lambda a: a[1]['volume'], x[1].items()))), d_dealer_2_dealer.items()))
interact_mean_bonds = list(map(lambda x: np.mean(
    list(map(lambda a: len(a[1]['bonds']), x[1].items()))), d_dealer_2_dealer.items()))


def show_statistic(_list, name):
    print('\n--------------------------------------')
    print(f'mean {name}: {np.mean(_list)}')
    print(f'std {name}: {np.std(_list)}')
    print(f'max {name}: {np.max(_list)}')
    print(f'min {name}: {np.min(_list)}')


show_statistic(interact_dealer, 'interact_dealer')
show_statistic(interact_mean_count, 'interact_mean_count')
show_statistic(interact_mean_volume, 'interact_mean_volume')
show_statistic(interact_mean_bonds, 'interact_mean_bonds')
