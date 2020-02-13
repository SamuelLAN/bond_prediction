import os
import numpy as np


def __get_volumes():
    cur_path = os.path.split(__file__)[0]
    root_dir = os.path.split(cur_path)[0]
    tmp_dir = os.path.join(root_dir, 'tmp')

    volume_path = os.path.join(tmp_dir, 'volume_of_dealer.txt')
    with open(volume_path, 'r') as f:
        content = f.readlines()
    data = list(map(lambda x: x.replace('[', '').replace(']', '').replace(',', '').replace("'", '').strip().split(' '),
                    content))
    data = list(map(lambda x: [int(x[0]), x[1], int(x[2])], data))
    return np.array(data, dtype=np.object)


def top_n(n):
    data = __get_volumes()
    return data[-int(n):, 1]


def by_range(minimum_volume, maximum_volume=None):
    data = __get_volumes()
    new_data = []
    for v in data:
        if v[-1] >= minimum_volume and (not maximum_volume or (v[-1] <= maximum_volume)):
            new_data.append(v[1])
    return new_data
