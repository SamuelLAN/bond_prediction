import os
import copy
import numpy as np
from gensim import corpora
from matplotlib import pyplot as plt
from config import path, date
from lib import utils

# dict_bond_id_topics = utils.load_json(os.path.join(path.DATA_ROOT_DIR, 'dict_bond_id_topics.json'))
dict_bond_id_topics = {}


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
    print(f'loading from dir {_dir_path} ...\n')

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


# data = utils.load_json(file_path)

def __2_one_hot(x, voc_size):
    return np.eye(voc_size)[x]


def __2_sum_one_hot(x, voc_size):
    return np.sum(__2_one_hot(x, voc_size), axis=0)


def __process(_data, remove_bond_list, dir_name, no_below, force_no_direction):
    doc_list = copy.deepcopy(_data)
    # doc_list = list(map(lambda x: list(map(lambda a: a[0], x)) if x else x, _data))

    def __remove_new_bonds(doc):
        doc = list(doc)
        for _bond_id in remove_bond_list:
            while [_bond_id, 'stc'] in doc:
                doc.remove([_bond_id, 'stc'])
            while [_bond_id, 'std'] in doc:
                doc.remove([_bond_id, 'std'])
            while [_bond_id, 'bfc'] in doc:
                doc.remove([_bond_id, 'bfc'])
            while [_bond_id, 'bfd'] in doc:
                doc.remove([_bond_id, 'bfd'])
        return doc

    doc_list = list(map(__remove_new_bonds, doc_list))

    print('generating dictionary ...')
    dictionary = corpora.Dictionary(list(map(lambda x: list(map(lambda a: a[0], x)) if x else x, doc_list)))
    original_bond_size = len(dictionary)

    if dict_bond_id_topics:
        def __filter_bond_not_in_topics(doc):
            return [x for x in doc if x[0] in dict_bond_id_topics]

        doc_list = list(map(__filter_bond_not_in_topics, doc_list))

        while [] in doc_list:
            doc_list.remove([])

        doc_list = list(map(lambda x: list(map(lambda a: list(map(lambda b: [b, a[1]], dict_bond_id_topics[a[0]])), x)), doc_list))

        doc_list = list(map(lambda x: list(np.vstack(x)), doc_list))

        indices_list = list(map(lambda x: list(map(lambda a: int(a[0]), x)), doc_list))

        voc_size = 100

    else:
        dictionary.filter_extremes(no_below=no_below)

        # dictionary size plus one unknown
        voc_size = len(dictionary)

        if voc_size < 5:
            return None, 0, 0

        print('converting to indices list\n')
        indices_list = list(map(lambda x: dictionary.doc2idx(x),
                                list(map(lambda x: list(map(lambda a: a[0], x)) if x else x, doc_list))
                                ))

        def __remove_few_transactions(_indices, _doc):
            len_indices = len(_indices) - 1
            while len_indices >= 0:
                if _indices[len_indices] == -1:
                    del _indices[len_indices]
                    del _doc[len_indices]
                len_indices -= 1

            return _indices, _doc

        for i, indices in enumerate(indices_list):
            indices_list[i], doc_list[i] = __remove_few_transactions(indices, doc_list[i])

        # # filter bonds that have few transactions
        # def __remove_bonds_with_few_transactions(_indices):
        #     while -1 in _indices:
        #         _indices.remove(-1)
        #     return _indices
        #
        # indices_list = list(map(__remove_bonds_with_few_transactions, indices_list))

    print(f'voc_size: {voc_size}\n')

    print(indices_list)

    print('converting to one hot vectors ...\n')
    if dir_name not in ['bonds_by_dealer', 'bonds_by_dealer_clients', 'bonds_by_dealer_dealers'] or force_no_direction:
        dates = list(map(lambda x: __2_sum_one_hot(x, voc_size), indices_list))

    else:
        dates = []
        length = len(indices_list)
        for i, indices in enumerate(indices_list):
            if i % 10 == 0:
                progress = float(i + 1) / length * 100.
                print('\rprogress: %.2f%%' % progress, end='')

            l = np.zeros((voc_size,))
            tmp_data = doc_list[i]

            for j, index in enumerate(indices):
                one_hot = __2_one_hot(index, voc_size)
                tmp_type = tmp_data[j][-1]
                if tmp_type[0] == 'b':
                    l += one_hot
                else:
                    l -= one_hot

            dates.append(l)

    print('finish processing \n')

    print(doc_list)
    print(voc_size)

    return dates, voc_size, original_bond_size


def cal_max_min_unit_for_hist(hist, num_unit, unit_is_integer=True):
    _max = np.max(hist)
    _min = np.min(hist)
    unit = float(_max - _min) / num_unit
    if unit_is_integer:
        if unit > 1:
            unit = np.ceil(unit)
            _max = int(_max) + unit - int(_max) % unit
        else:
            unit = 1
            _max = int(_max) + 1
    else:
        pass

    return _max + unit, _min, unit


def plot(dates, voc_size, original_bond_size, name, save_dir, dealer_index, size_times, no_below):
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

    if not d:
        return

    l = list(zip(*l))
    # plt.scatter(l[0], l[1], color='blue', s=1)

    print('plotting ... \n')

    rets = {}

    plt.figure(figsize=(20., 20 * 4.8 / 10.4))
    X = []
    Y = []

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

        X += v[0]
        Y += v[1]

        p = plt.scatter(v[0], v[1], color=color, s=s, label=_type)
        # rets[_type] = p

    # plt_list = [v for k, v in rets.items()]

    if dict_bond_id_topics:
        title = f'dealer_{dealer_index} of {name} (topic_size: {voc_size}, original_bond_size: {original_bond_size})'
    else:
        title = f'dealer_{dealer_index} of {name} (bond_size: {voc_size}, no_below: {no_below}, original_bond_size: {original_bond_size})'

    plt.title(title)
    plt.xlabel('dates (only weekdays from Jan 2nd to Dec 31th)')
    plt.ylabel('bond index')

    max_y, min_y, unit_y = cal_max_min_unit_for_hist(Y, 25)
    max_x, min_x, unit_x = cal_max_min_unit_for_hist(X, 25)

    plt.xticks(np.arange(min_x, max_x, unit_x))
    plt.yticks(np.arange(min_y, max_y, unit_y))
    plt.legend()

    save_path = os.path.join(save_dir, f'dealer_{dealer_index}_no_below_{no_below}.png')
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


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
dealer_index = '493'
no_below = 10
size_times = 1
force_no_direction = True

dir_path = os.path.join(path.PREDICTION_DATE_DIR, dir_name, '2015', f'dealer_{dealer_index}')
name = dir_name.replace('bonds_by_dealer_', '').replace('bonds_by_dealer', '')
name = name_map[name]

save_dir = os.path.join(path.ROOT_DIR, 'runtime', 'test')
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

path_dict_bond_id_offering_date = os.path.join(path.DATA_ROOT_DIR, 'dict_bond_id_offering_date.json')
dict_bond_id_offering_date = utils.load_json(path_dict_bond_id_offering_date)
remove_bond_list = []

bound_date = '2014-06-01'
bound_timestamp = utils.date_2_timestamp(bound_date)
for bond_id, offering_date in dict_bond_id_offering_date.items():
    if offering_date[:2] == '19':
        continue
    elif offering_date[:1] != '2':
        remove_bond_list.append(bond_id)
        continue

    offering_date = offering_date.replace('-00', '-01')

    offering_timestamp = utils.date_2_timestamp(offering_date)
    if offering_timestamp >= bound_timestamp:
        remove_bond_list.append(bond_id)

data = __load_dir(dir_path)
dates, voc_size, original_bond_size = __process(data, remove_bond_list, dir_name, no_below, force_no_direction)
plot(dates, voc_size, original_bond_size, name, save_dir, dealer_index, size_times, no_below)

print('done')
