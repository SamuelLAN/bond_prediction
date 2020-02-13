import os
import numpy as np
from gensim import corpora
from matplotlib import pyplot as plt
from config import path, date
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

dir_name = 'bonds_by_dealer'
dealer_index = '1347'
no_below = 20
size_times = 1
force_no_direction = False


dir_path = os.path.join(path.PREDICTION_DATE_DIR, dir_name, '2015', f'dealer_{dealer_index}')
name = dir_name.replace('bonds_by_dealer_', '').replace('bonds_by_dealer', '')
name = name_map[name]


def __load_dir(_dir_path):
    """
    Load all the data in "dir_path", and complement the data in the dates that no transaction happened
    :return
        data: (list)
        e.g. [ # include transactions happen in many days
            ['bond_a', 'bond_b', ...], # represent transaction happen in one day
            ['bond_a', 'bond_b', ...],
            ...
        ]
    """
    _data = []

    # load the date list
    date_list = os.listdir(_dir_path)
    date_list.sort()

    # generate a date dict so that we can check whether there is transaction happens in that date
    date_dict = utils.list_2_dict(date_list)

    # find out the start and end date of all the transactions
    start_date = date_list[0][len('doc_'): -len('.json')]
    end_date = date_list[-1][len('doc_'): -len('.json')]

    # covert the date to timestamp
    cur_timestamp = utils.date_2_timestamp(start_date)
    end_timestamp = utils.date_2_timestamp(end_date) + 86000

    # traverse all the date between the start date and the end date, but skip the holidays
    while cur_timestamp < end_timestamp:
        _date = utils.timestamp_2_date(cur_timestamp)
        file_name = f'doc_{_date}.json'

        # check if there is any transaction
        if file_name in date_dict:
            file_path = os.path.join(_dir_path, file_name)

            # remove nan in doc
            tmp_doc = utils.load_json(file_path)
            tmp_doc = list(map(lambda x: x if isinstance(x[0], str) else '', tmp_doc))
            while '' in tmp_doc:
                tmp_doc.remove('')
            _data.append(tmp_doc)

        # if it is holidays, then skip it
        elif date.is_holiday(_date):
            pass

        # if no transaction happens in that date
        else:
            _data.append([])

        # move to the next day
        cur_timestamp += 86400

    return _data


print(f'loading from dir {dir_path} ...\n')
data = __load_dir(dir_path)
# data = utils.load_json(file_path)

doc_list = list(map(lambda x: list(map(lambda a: a[0], x)) if x else x, data))

print('generating dictionary ...')
dictionary = corpora.Dictionary(doc_list)
dictionary.filter_extremes(no_below=no_below)

# dictionary size plus one unknown
voc_size = len(dictionary) + 1

print(f'voc_size: {voc_size}\n')

print('converting to indices list\n')
indices_list = list(map(lambda x: dictionary.doc2idx(x), doc_list))


def __2_one_hot(x):
    return np.eye(voc_size)[x]


def __2_sum_one_hot(x, max_0_val=None):
    l = np.sum(__2_one_hot(x), axis=0)
    if max_0_val:
        l[l > 0] = max_0_val
    return l


print('converting to one hot vectors ...\n')
if dir_name not in ['bonds_by_dealer', 'bonds_by_dealer_clients', 'bonds_by_dealer_dealers'] or force_no_direction:
    dates = list(map(__2_sum_one_hot, indices_list))

else:
    dates = []
    length = len(indices_list)
    for i, indices in enumerate(indices_list):
        if i % 10 == 0:
            progress = float(i + 1) / length * 100.
            print('\rprogress: %.2f%%' % progress, end='')

        l = np.zeros((voc_size,))
        tmp_data = data[i]

        for j, index in enumerate(indices):
            one_hot = __2_one_hot(index)
            tmp_type = tmp_data[j][-1]
            if tmp_type[0] == 'b':
                l += one_hot
            else:
                l -= one_hot

        dates.append(l)

print('finish processing \n')

print(doc_list)
print(voc_size)

print('preparing plotting data ...\n')
d = {}
l = []
for i, v in enumerate(dates):
    where = list(map(lambda x: [i, x[0]], np.argwhere(v != 0)))
    val = list(map(lambda x: x[0], v[np.argwhere(v != 0)]))

    l += where

    for j, count in enumerate(val):
        if str(count) not in d:
            d[str(count)] = []
        d[str(count)].append(where[j])

l = list(zip(*l))
# plt.scatter(l[0], l[1], color='blue', s=1)

print('plotting ... \n')

rets = {}

for k, v in d.items():
    # if 'buy', then 'blue'; if 'sell', then 'red'
    color = 'blue' if float(k) >= 0 else 'red'
    _type = 'buy' if float(k) >= 0 else 'sell'

    if name in ['StC', 'StD']:
        color = 'red'
        _type = 'sell'

    s = int(abs(float(k))) * size_times
    v = list(zip(*v))

    if _type not in rets:
        rets[_type] = True
    else:
        _type = None
    p = plt.scatter(v[0], v[1], color=color, s=s, label=_type)
    # rets[_type] = p

print('done')

# plt_list = [v for k, v in rets.items()]

plt.title(f'dealer_{dealer_index} of {name} (bond_size: {voc_size}, no_below: {no_below})')
plt.xlabel('dates (only weekdays from Jan 2nd to Dec 31th)')
plt.ylabel('bond index')
plt.legend()
plt.show()
