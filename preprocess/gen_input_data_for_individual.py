import os

cur_path = os.path.split(os.path.abspath(__file__))[0]

import numpy as np
from gensim import corpora
from six.moves import cPickle as pickle
from config import path, date
from lib import utils


def zero_pad(len_bonds, with_day_off=True):
    total_len = len_bonds + 3 if with_day_off else len_bonds + 2
    return np.zeros((total_len,))


def __token(_len):
    return np.arange(_len)


def __start_token_mask(shape, with_day_off=True):
    x = np.zeros(shape)
    start_index = -3 if with_day_off else -2
    x[start_index] = 1
    return x


def __end_token_mask(shape, with_day_off=True):
    x = np.zeros(shape)
    end_index = -2 if with_day_off else -1
    x[end_index] = 1
    return x


def __convert_2_zero_one(ar_x):
    ar_x = np.sum(ar_x, axis=0)
    ar_x[ar_x > 0] = 1
    ar_x[ar_x <= 0] = -1
    return ar_x


def __sample_list(array, window=1, start=0, end=0, start_token=None, end_token=None,
                  convert_fn=None, longest_len=None):
    x = []
    for i, v in enumerate(array[start: len(array) - window - end + 1]):
        tmp_x = array[i: i + window]
        if not isinstance(start_token, type(None)):
            tmp_x = np.vstack([start_token, tmp_x])
        if not isinstance(end_token, type(None)):
            tmp_x = np.vstack([tmp_x, end_token])
        if longest_len:
            tmp_x = np.vstack([tmp_x, np.zeros((longest_len - len(tmp_x), tmp_x.shape[-1]))])
        if not isinstance(convert_fn, type(None)):
            tmp_x = convert_fn(tmp_x)
        x.append(tmp_x)

    return x


def __change_mask(buy_sell_plan, _mask, _bond_idx, len_bonds, _dict_date_2_input_m_index, _date, _trace_type, value=1):
    date_index = _dict_date_2_input_m_index[_date]
    if buy_sell_plan == 2:
        _mask[date_index, _bond_idx + len_bonds] += value
    elif buy_sell_plan == 1 and _trace_type[0].lower() == 's':
        _mask[date_index, _bond_idx] -= value
    else:
        _mask[date_index, _bond_idx] += value
    return _mask


def __generate_date_structure(len_bonds, start_date='2015-01-02', end_date='2015-12-31',
                              with_day_off=True, buy_sell_plan=2):
    """
    return:
        date_matrix: np.array, shape: (date_num, len_bonds + 3) or (date_num, len_bonds * 2 + 3),
                    values are all ones. Therefore, a mask for the input is needed.
                    "+ 3" is because the last 3 tokens are for <start> <end> <day-off>
                    <pad> uses all zeros, thus, it does not need a place in vocabulary
        date_mask: np.array, shape: (date_num, len_bonds + 3) or (date_num, len_bonds * 2 + 3),
                    values are all zeros. The mask would be changed when traversing the transaction history.
        dict_date_2_input_m_index: dict, map the date to the index of date_matrix
    """
    dict_date_2_input_m_index = {}

    # convert timestamp
    start_timestamp = utils.date_2_timestamp(start_date)
    end_timestamp = utils.date_2_timestamp(end_date, True)

    # temporary variables
    l = []
    cur_timestamp = start_timestamp

    # generate the dict_date_2_input_m_index and calculate the date_num
    while cur_timestamp <= end_timestamp:
        _date = utils.timestamp_2_date(cur_timestamp)
        cur_timestamp += 86400

        if date.is_holiday(_date):
            if with_day_off:
                l.append(0)
            continue

        dict_date_2_input_m_index[_date] = len(l)
        l.append(1)

    # calculate the shape
    date_num = len(l)
    extra_token_num = 3 if with_day_off else 2
    len_bonds = len_bonds * 2 + extra_token_num if buy_sell_plan == 2 else len_bonds + extra_token_num

    # generate variables
    date_matrix = np.ones((date_num, 1)) * np.arange(len_bonds)
    date_mask = np.zeros((date_num, len_bonds)) * np.arange(len_bonds)
    # change the value in day off pos to be 1
    date_mask[np.where(np.array(l) == 0)[0], -1] = 1

    return date_matrix, date_mask, dict_date_2_input_m_index


def gen_inputs(group_file_path, group_index, input_time_steps_list, output_time_steps_list,
               with_day_off=True, buy_sell_plan=2, use_volume=False,
               save_path='', split_ratio=0.9, is_train=True):
    d_dealer_index_2_group_label = utils.load_json(group_file_path)
    # d_dealer_index_2_trace_list = utils.load_pkl(os.path.join(path.ROOT_DIR, 'runtime', 'tmp_d_dealers.pkl'))

    d_dealer_for_gen_input = utils.load_pkl(
        os.path.join(path.RUNTIME_DIR, 'd_dealer_for_gen_input_with_no_below_50_25_10.pkl'))

    _individual_index = ''
    if isinstance(group_index, int):
        tmp_list = [dealer_index for dealer_index, group_label in d_dealer_index_2_group_label.items()
                    if group_label == group_index]
        print(f'len_group_member: {len(tmp_list)}')
    else:
        print(f'len_group_member: 1')
        _individual_index = group_index[1:]

    # get total trace list
    train_trace_list = []
    test_trace_list = []
    for dealer_index, val in d_dealer_for_gen_input.items():
        if dealer_index not in d_dealer_index_2_group_label or _individual_index != dealer_index:
            continue
        trace_list = val['trace_list']
        trace_list.sort(key=lambda x: x[-1])

        num_samples = len(trace_list) - max(input_time_steps_list) - max(output_time_steps_list)
        split_index = int(num_samples * split_ratio + max(input_time_steps_list))
        train_trace_list += trace_list[: split_index]
        test_trace_list += trace_list[split_index:]

        # print(dealer_index, len(trace_list), trace_list)

    train_trace_list.sort(key=lambda x: x[-1])

    # get dictionary
    train_doc_list = [list(map(lambda x: x[0], train_trace_list))]
    dictionary = corpora.Dictionary(train_doc_list)
    len_bonds = len(dictionary)
    print(f'total bond num (group {group_index}): {len_bonds}')

    X = []
    X_mask = []
    Y = []

    for dealer_index, val in d_dealer_for_gen_input.items():
        if dealer_index not in d_dealer_index_2_group_label or _individual_index != dealer_index:
            continue

        # filter bonds that only appear in test set

        trace_list = val['trace_list']
        num_samples = len(trace_list) - max(input_time_steps_list) - max(output_time_steps_list)
        split_index = int(num_samples * split_ratio + max(input_time_steps_list))

        if is_train:
            trace_list = trace_list[:split_index]
        else:
            trace_list = trace_list[split_index:]
            trace_list = [v for v in trace_list if dictionary.doc2idx([v[0]])[0] != -1]
        trace_list.sort(key=lambda x: x[-1])

        start_date = trace_list[0][-1]
        end_date = trace_list[-1][-1]

        # Format the data in date structure
        date_matrix, date_mask, dict_date_2_input_m_index = __generate_date_structure(len_bonds,
                                                                                      start_date,
                                                                                      end_date,
                                                                                      with_day_off,
                                                                                      buy_sell_plan)

        # according to the transaction history, fill the data into date structure
        for i, trace in enumerate(trace_list):
            bond_id = trace[0]
            volume = trace[1]
            _date = trace[-1]
            trace_type = trace[2]
            bond_index = dictionary.doc2idx([bond_id])[0]

            value = 1 if not use_volume else np.log10(volume)
            if _date not in dict_date_2_input_m_index:
                continue

            date_mask = __change_mask(
                buy_sell_plan, date_mask, bond_index, len_bonds, dict_date_2_input_m_index, _date, trace_type, value)

        # sample the data
        longest_input_length = max(input_time_steps_list) + 2
        for input_time_steps in input_time_steps_list:
            for output_time_steps in output_time_steps_list:
                # input_list = __sample_list(date_matrix, input_time_steps, 0, output_time_steps,
                #                            __token(date_matrix.shape[-1]), __token(date_matrix.shape[-1]))
                input_mask_list = __sample_list(date_mask, input_time_steps, 0, output_time_steps,
                                                __start_token_mask(date_mask.shape[1:], with_day_off),
                                                __end_token_mask(date_mask.shape[1:], with_day_off),
                                                longest_len=longest_input_length)

                convert_fn = __convert_2_zero_one if buy_sell_plan in [0, 2] else None
                output_list = __sample_list(date_mask, output_time_steps, input_time_steps, 0, convert_fn=convert_fn)

                if len(input_mask_list) != len(output_list):
                    continue

                # ...
                # X += input_list
                X_mask += input_mask_list
                Y += output_list

        # d_dealer_index_2_trace_list_ordered[dealer_index] = [date_matrix, date_mask]

    if save_path:
        del d_dealer_for_gen_input
        del d_dealer_index_2_group_label
        del X

        X_mask = np.asarray(X_mask, dtype=np.int32)
        Y = np.asarray(Y, dtype=np.int32)

        print('\n------------------------------')
        # print(len(X))
        print(X_mask.shape)
        print(Y.shape)

        len_X = len(X_mask)
        num_files = int(np.ceil(len_X / 2000.))
        for i in range(num_files):
            start_index = i * 2000
            end_index = (i + 1) * 2000
            utils.write_pkl(save_path + f'_{i}.pkl', [X_mask[start_index: end_index], Y[start_index: end_index]])

    # return d_dealer_index_2_trace_list_ordered
    return X_mask, Y


group_name = 'group_Spectral_Clustering_filter_lower_5_with_model_input_features.json'
param_name = 'no_day_off_no_distinguish_buy_sell_use_transaction_count'
group_index = 4
is_train = True
individual_index = '3260'
if individual_index:
    group_index = f'i{individual_index}'

group_file_path = os.path.join(path.ROOT_DIR, 'group', group_name)
dataset_suffix = 'train' if is_train else 'test'

save_dir_path = os.path.join(path.DATA_ROOT_DIR, 'inputs', group_name.split('.')[0])
if not os.path.exists(save_dir_path):
    os.mkdir(save_dir_path)
save_dir_path = os.path.join(save_dir_path, param_name)
if not os.path.exists(save_dir_path):
    os.mkdir(save_dir_path)
save_dir_path = os.path.join(save_dir_path, f'group_{group_index}_no_below_50_25_10_g_minus_1_1_{dataset_suffix}')
if not os.path.exists(save_dir_path):
    os.mkdir(save_dir_path)
save_file_path = os.path.join(save_dir_path, 'data')

gen_inputs(group_file_path, group_index, [5, 10, 15], [2], False, 0, False,
           save_path=save_file_path, is_train=is_train)
