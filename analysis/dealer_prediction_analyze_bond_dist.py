import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from functools import reduce
from lib import utils

cache_path = utils.get_relative_dir('runtime', 'cache', f'l_bonds.pkl')
l_bonds = utils.load_pkl(cache_path)

l_bonds.sort(key=lambda x: x['trace_count'], reverse=True)

total_trace_count = sum(list(map(lambda x: x['trace_count'], l_bonds)))
total_volume = sum(list(map(lambda x: x['volume'], l_bonds)))
total_trace_days = sum(list(map(lambda x: x['trace_date_count'], l_bonds)))
all_dealer_set = reduce(lambda a, b: a.union(b['dealer_set']), l_bonds, set())

num_bonds = 0
subset_trace_count = 0
subset_volume = 0
subset_trace_days = 0
subset_dealers = set()

d_show = {}

for i, v in enumerate(l_bonds):
    subset_trace_count += v['trace_count']
    subset_volume += v['volume']
    subset_trace_days += v['trace_date_count']
    subset_dealers = subset_dealers.union(v['dealer_set'])

    trace_count_percentage = subset_trace_count / total_trace_count * 100.
    volume_percentage = subset_volume / total_volume * 100.
    trace_days_percentage = subset_trace_days / total_trace_days * 100.
    dealer_count_percentage = len(subset_dealers) / len(all_dealer_set) * 100.

    for threshold in list(range(50, 105, 5)) + [98, 99, 99.99]:

        if trace_count_percentage > threshold and threshold not in d_show:
            d_show[threshold] = True
            print('\n----------------------------------------------')
            print(f'num_bond: {i + 1}')
            print(f'trace_count: {subset_trace_count} / {round(trace_count_percentage, 2)}% ')
            print(f'volume: {subset_volume} / {round(volume_percentage, 2)}% ')
            print(f'trace_days: {subset_trace_days} / {round(trace_days_percentage, 2)}% ')
            print(f'dealer_count: {len(subset_dealers)} / {round(dealer_count_percentage, 2)}% ')

# histogram for total transaction count
trace_count_list = list(map(lambda x: x['trace_count'], l_bonds[:1600]))
trace_days_list = list(map(lambda x: x['trace_date_count'], l_bonds[:1600]))
# trace_days_trace_count_list = list(map(lambda x: [x['trace_date_count'], np.log10(x['trace_count'])], l_bonds))
trace_days_trace_count_list = list(map(lambda x: [x['trace_date_count'], x['trace_count']], l_bonds[:1600]))
dealer_count_list = list(map(lambda x: len(x['dealer_set']), l_bonds[:1600]))

plt.figure(figsize=(18., 18 * 4.8 / 10.4))
plt.hist(trace_count_list, bins=list(range(0, 2500, 500)) + list(range(2000, 11000, 1000)) + [20000, 30000],
         edgecolor='white')
plt.title('Histogram for transaction count per bond (first 1600 bonds)', fontsize=25)
plt.xlabel('transaction count per bond', fontsize=20)
plt.ylabel('count of bonds', fontsize=20)

plt.xticks(np.arange(0, 30000, 3000), fontsize=14)
plt.yticks(np.arange(0, 5500, 500), fontsize=14)

plt.savefig(utils.get_relative_file('runtime', 'analysis', 'dealer_prediction', 'hist_trace_count_first_1600.png'),
            dpi=300)
plt.show()
plt.close()

plt.figure(figsize=(18., 18 * 4.8 / 10.4))
plt.hist(trace_days_list, bins=list(range(0, 260, 10)), edgecolor='white')
plt.title('Histogram for transaction days per bond (first 1600 bonds)', fontsize=25)
plt.xlabel('transaction days per bond', fontsize=20)
plt.ylabel('count of bonds', fontsize=20)

plt.xticks(list(range(0, 280, 20)), fontsize=14)
plt.yticks(np.arange(0, 700, 100), fontsize=14)

plt.savefig(utils.get_relative_file('runtime', 'analysis', 'dealer_prediction', 'hist_trace_day_count_first_1600.png'),
            dpi=300)
plt.show()
plt.close()

X, Y = list(zip(*trace_days_trace_count_list))
plt.figure(figsize=(18., 18 * 4.8 / 10.4))
plt.scatter(X, Y, color='red', s=1, label='bonds')
plt.yscale('log')
plt.title('Plot of transaction days vs transaction count (first 1600 bonds)', fontsize=25)
plt.xlabel('transaction days', fontsize=20)
plt.ylabel('transaction count', fontsize=20)
plt.xticks(list(range(0, 280, 20)), fontsize=14)

plt.savefig(
    utils.get_relative_file('runtime', 'analysis', 'dealer_prediction', 'plot_trace_day_vs_trace_count_first_1600.png'),
    dpi=300)
plt.show()
plt.close()

plt.figure(figsize=(18., 18 * 4.8 / 10.4))
plt.hist(dealer_count_list, bins=list(range(0, 360, 15)), edgecolor='white')
plt.title('Histogram for distinct dealer count days per bond (first 1600 bonds)', fontsize=25)
plt.xlabel('distinct dealer count per bond', fontsize=20)
plt.ylabel('count of bonds', fontsize=20)

plt.xticks(list(range(0, 360, 15)), fontsize=14)
plt.yticks(np.arange(0, 1750, 150), fontsize=14)

plt.savefig(
    utils.get_relative_file('runtime', 'analysis', 'dealer_prediction', 'hist_distinct_dealer_count_first_1600.png'),
    dpi=300)
plt.show()
plt.close()

plt.figure(figsize=(18., 18 * 4.8 / 10.4))
plt.hist2d(trace_count_list, dealer_count_list, bins=20, edgecolor='white', cmap='Blues')
plt.colorbar()
plt.title('Histogram for transaction count and distinct dealer count days per bond (first 1600 bonds)', fontsize=25)
plt.xlabel('transaction count per bond', fontsize=20)
plt.ylabel('distinct dealer count per bond', fontsize=20)

plt.savefig(utils.get_relative_file('runtime', 'analysis', 'dealer_prediction',
                                    'hist_trace_count_and_distinct_dealer_count_first_1600_2D.png'), dpi=300)
plt.show()
plt.close()

plt.figure(figsize=(18., 18 * 4.8 / 10.4))
ax = plt.axes(projection='3d')

hist, xedges, yedges = np.histogram2d(trace_count_list, dealer_count_list, bins=20)
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0
dx = xedges[1] - xedges[0]
dy = yedges[1] - yedges[0]
dz = hist.ravel()
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

plt.title('Histogram for transaction count and distinct dealer count days per bond (first 1600 bonds)', fontsize=25)
ax.set_xlabel('transaction count per bond', fontsize=20)
ax.set_ylabel('distinct dealer count per bond', fontsize=20)
ax.set_zlabel('count of bonds', fontsize=20)
plt.savefig(utils.get_relative_file('runtime', 'analysis', 'dealer_prediction',
                                    'hist_trace_count_and_distinct_dealer_count_first_1600_3D.png'), dpi=300)
plt.show()
plt.close()

plt.figure(figsize=(18., 18 * 4.8 / 10.4))
plt.hist2d(trace_count_list, trace_days_list, bins=20, edgecolor='white', cmap='Blues')
plt.colorbar()
plt.title('Histogram for transaction count and transaction days per bond (first 1600 bonds)', fontsize=25)
plt.xlabel('transaction count per bond', fontsize=20)
plt.ylabel('transaction days per bond', fontsize=20)

plt.savefig(utils.get_relative_file('runtime', 'analysis', 'dealer_prediction',
                                    'hist_trace_count_and_trace_days_first_1600_2D.png'), dpi=300)
plt.show()
plt.close()

plt.figure(figsize=(18., 18 * 4.8 / 10.4))
ax = plt.axes(projection='3d')

hist, xedges, yedges = np.histogram2d(trace_count_list, trace_days_list, bins=20)
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0
dx = xedges[1] - xedges[0]
dy = yedges[1] - yedges[0]
dz = hist.ravel()
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

plt.title('Histogram for transaction count and transaction days per bond (first 1600 bonds)', fontsize=25)
ax.set_xlabel('transaction count per bond', fontsize=20)
ax.set_ylabel('transaction days per bond', fontsize=20)
ax.set_zlabel('count of bonds', fontsize=20)
plt.savefig(utils.get_relative_file('runtime', 'analysis', 'dealer_prediction',
                                    'hist_trace_count_and_trace_days_first_1600_3D.png'), dpi=300)
plt.show()
plt.close()

plt.figure(figsize=(18., 18 * 4.8 / 10.4))
ax = plt.axes(projection='3d')
ax.scatter(trace_days_list, dealer_count_list, trace_count_list, cmap='Blues', linewidth=0.5)
# plt.colorbar()
plt.title('Plot for trace days, dealer count, and trace days per bond (first 1600 bonds)', fontsize=25)
ax.set_xlabel('transaction days per bond', fontsize=20)
ax.set_ylabel('distinct dealer count per bond', fontsize=20)
ax.set_zlabel('transaction count per bond', fontsize=20)
plt.savefig(utils.get_relative_file('runtime', 'analysis', 'dealer_prediction',
                                    'plot_x_trace_day_y_dealer_count_z_trace_count_first_1600_3D.png'), dpi=300)
plt.show()
plt.close()

# ----------------------------------------------
# num_bond: 743
# trace_count: 2346110 / 50.0%
# volume: 795704877719.75 / 32.37%
# trace_days: 162589 / 19.0%
# dealer_count: 1115 / 93.07%
#
# ----------------------------------------------
# num_bond: 911
# trace_count: 2581255 / 55.01%
# volume: 908658880205.28 / 36.97%
# trace_days: 197765 / 23.11%
# dealer_count: 1123 / 93.74%
#
# ----------------------------------------------
# num_bond: 1106
# trace_count: 2816299 / 60.02%
# volume: 1038857605262.53 / 42.27%
# trace_days: 238083 / 27.83%
# dealer_count: 1130 / 94.32%
#
# ----------------------------------------------
# num_bond: 1338
# trace_count: 3050373 / 65.01%
# volume: 1169589331482.53 / 47.59%
# trace_days: 284054 / 33.2%
# dealer_count: 1140 / 95.16%
#
# ----------------------------------------------
# num_bond: 1616
# trace_count: 3284727 / 70.01%
# volume: 1311760059527.3801 / 53.37%
# trace_days: 336211 / 39.29%
# dealer_count: 1149 / 95.91%
#
# ----------------------------------------------
# num_bond: 1949
# trace_count: 3519484 / 75.01%
# volume: 1460545539665.76 / 59.42%
# trace_days: 395907 / 46.27%
# dealer_count: 1157 / 96.58%
#
# ----------------------------------------------
# num_bond: 2357
# trace_count: 3754021 / 80.01%
# volume: 1613836206620.51 / 65.66%
# trace_days: 463625 / 54.19%
# dealer_count: 1167 / 97.41%
#
# ----------------------------------------------
# num_bond: 2876
# trace_count: 3988551 / 85.01%
# volume: 1786100924986.76 / 72.67%
# trace_days: 540740 / 63.2%
# dealer_count: 1177 / 98.25%
#
# ----------------------------------------------
# num_bond: 3577
# trace_count: 4223146 / 90.01%
# volume: 1972248484074.7202 / 80.24%
# trace_days: 627150 / 73.3%
# dealer_count: 1179 / 98.41%
#
# ----------------------------------------------
# num_bond: 4641
# trace_count: 4457603 / 95.0%
# volume: 2180189594756.1602 / 88.7%
# trace_days: 729722 / 85.29%
# dealer_count: 1185 / 98.91%
#
# ----------------------------------------------
# num_bond: 5738
# trace_count: 4598270 / 98.0%
# volume: 2336945612400.64 / 95.08%
# trace_days: 801875 / 93.72%
# dealer_count: 1191 / 99.42%
#
# ----------------------------------------------
# num_bond: 6346
# trace_count: 4645183 / 99.0%
# volume: 2392911088386.3604 / 97.36%
# trace_days: 827905 / 96.76%
# dealer_count: 1192 / 99.5%
#
# ----------------------------------------------
# num_bond: 7667
# trace_count: 4691640 / 99.99%
# volume: 2457409619688.7305 / 99.98%
# trace_days: 855327 / 99.97%
# dealer_count: 1198 / 100.0%
