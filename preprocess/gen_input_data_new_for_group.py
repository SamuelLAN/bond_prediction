import os
import copy

cur_path = os.path.split(os.path.abspath(__file__))[0]

import numpy as np
from gensim import corpora
from six.moves import cPickle as pickle
from config import path, date
from lib import utils

freq_level = 'more'


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


def __merge_dates(ar_x):
    ar_x = np.sum(ar_x, axis=0)
    return ar_x


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
        x.append(np.asarray(tmp_x, dtype=np.int32))

    return x


def __change_mask(buy_sell_plan, _mask, _bond_idx, len_bonds, _dict_date_2_input_m_index, _date, _trace_type, value=1,
                  only_buy=False, only_sell=False):
    date_index = _dict_date_2_input_m_index[_date]

    if only_buy and _trace_type[0].lower() == 's':
        return _mask

    if only_sell and _trace_type[0].lower() == 'b':
        return _mask

    if buy_sell_plan == 2:
        if _trace_type[0].lower() == 'b':
            _mask[date_index, _bond_idx] += value
        else:
            _mask[date_index, _bond_idx + len_bonds] += value
        # if _trace_type[0].lower() == 's':
        #     _mask[date_index, _bond_idx + len_bonds] += value
        # else:
        #     _mask[date_index, _bond_idx] += value
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
    # date_matrix = np.ones((date_num, 1), dtype=np.int32) * np.arange(len_bonds)
    date_mask = np.zeros((date_num, len_bonds), dtype=np.int32) * np.arange(len_bonds)
    # change the value in day off pos to be 1
    date_mask[np.where(np.array(l) == 0)[0], -1] = 1

    return None, date_mask, dict_date_2_input_m_index


def get_x_y_for_one_dealer(traces, dictionary, _input_windows, _output_windows, pre_traces=None,
                           _with_day_off=False, _buy_sell_plan=0, _use_volume=False, only_buy_y=False, only_sell_y=False):
    # get useful variables
    start_date = traces[0][-1]
    end_date = traces[-1][-1]
    len_bonds = len(dictionary)
    max_input_time_step = max(_input_windows)
    sample_start_index = 0

    # for validation or test set
    if not isinstance(pre_traces, type(None)):
        sample_start_index = max_input_time_step

        start_timestamp = utils.date_2_timestamp(start_date)
        weekday_num = max_input_time_step
        while weekday_num > 0:
            start_timestamp -= 86400
            if date.is_holiday_timestamp(start_timestamp):
                continue
            weekday_num -= 1

        start_date = utils.timestamp_2_date(start_timestamp)
        pre_traces = list(filter(lambda x: x[-1] >= start_date, pre_traces))
        traces += pre_traces

    # Format the data in date matrix
    _, date_mask, dict_date_2_input_m_index = __generate_date_structure(len_bonds, start_date, end_date,
                                                                        _with_day_off, _buy_sell_plan)

    date_mask_for_y = copy.deepcopy(date_mask)

    # according to the transaction history, fill the data into date structure
    for i, trace in enumerate(traces):
        # check if it is in holidays
        _date = trace[-1]
        if _date not in dict_date_2_input_m_index:
            continue

        # get values
        bond_id = trace[0]
        volume = trace[1]
        trace_type = trace[2]
        bond_index = dictionary.doc2idx([bond_id])[0]

        # check if use volume for the value of input elements
        value = 1 if not _use_volume else np.log10(volume)

        # change the date_mask on the day the transaction happened
        date_mask = __change_mask(
            _buy_sell_plan, date_mask, bond_index, len_bonds, dict_date_2_input_m_index, _date, trace_type, value)

        date_mask_for_y = __change_mask(
            _buy_sell_plan, date_mask_for_y, bond_index, len_bonds, dict_date_2_input_m_index, _date, trace_type, value,
            only_buy=only_buy_y, only_sell=only_sell_y)

    X = []
    y = []

    # sample the data
    longest_input_length = max_input_time_step + 2  # due to adding start and end token
    for input_time_steps in _input_windows:
        for output_time_steps in _output_windows:
            # calculate the start index for sampling inputs and outputs
            start_input_index = 0 if sample_start_index == 0 else sample_start_index - input_time_steps
            start_output_index = input_time_steps if sample_start_index == 0 else sample_start_index

            # get input mask
            input_mask_list = __sample_list(date_mask,
                                            window=input_time_steps,
                                            start=start_input_index,
                                            end=output_time_steps,
                                            start_token=__start_token_mask(date_mask.shape[1:], _with_day_off),
                                            end_token=__end_token_mask(date_mask.shape[1:], _with_day_off),
                                            longest_len=longest_input_length)

            # get ground truth
            convert_fn = __convert_2_zero_one if _buy_sell_plan in [0, 2] else None
            # convert_fn = None
            output_list = __sample_list(date_mask_for_y,
                                        window=output_time_steps,
                                        start=start_output_index,
                                        end=0,
                                        convert_fn=convert_fn)

            # check input and ground truth
            if len(input_mask_list) != len(output_list):
                continue

            # ...
            # X += input_list
            X += input_mask_list
            y += output_list

    return X, y


def load_d_dealers_after_filtering(_trace_suffix, use_cache=True):
    print('\nloading d_dealers_2_traces ... ')

    cache_path = utils.get_relative_dir('cache', 'filtered_d_dealers.pkl', root=path.RUNTIME_DIR)
    if os.path.exists(cache_path) and use_cache:
        return utils.load_pkl(cache_path)

    # load trace data
    train_d_dealers = utils.load_json(os.path.join(path.D_DEALERS_TRACE_DIR, f'train_{_trace_suffix}'))
    test_d_dealers = utils.load_json(os.path.join(path.D_DEALERS_TRACE_DIR, f'test_{_trace_suffix}'))

    # remove clients, we do not predict clients' behaviors
    if '0' in train_d_dealers:
        del train_d_dealers['0']
    if '99999' in train_d_dealers:
        del train_d_dealers['99999']
    if '0' in test_d_dealers:
        del test_d_dealers['0']
    if '99999' in test_d_dealers:
        del test_d_dealers['99999']

    new_train_d_dealers = {}
    new_test_d_dealers = {}

    length = len(train_d_dealers)
    count = 0

    for _dealer_index, train_traces in train_d_dealers.items():
        # output progress
        count += 1
        if count % 2 == 0:
            progress = float(count) / length * 100.
            print('\rprogress: %.2f%% ' % progress, end='')

        test_traces = test_d_dealers[_dealer_index]

        # calculate the total transaction count
        train_trace_count = len(train_traces)

        # filter bonds whose trade frequency is low
        if train_trace_count < 1000:
            continue

        # set filter boundaries according to the level of transaction count
        if train_trace_count > 100000:
            no_below = 45
        elif train_trace_count > 10000:
            no_below = 22
        else:
            no_below = 8

        # arrange the bonds according to their dates of transaction, for filtering purpose
        d_train_date_2_bonds = {}
        for v in train_traces:
            bond_id = v[0]
            trade_date = v[-1]

            if trade_date not in d_train_date_2_bonds:
                d_train_date_2_bonds[trade_date] = []
            d_train_date_2_bonds[trade_date].append(bond_id)

        # construct doc list for transactions
        train_doc_list = list(map(lambda x: x[1], d_train_date_2_bonds.items()))

        # construct dictionary for bonds
        train_dictionary = corpora.Dictionary(train_doc_list)
        # filter bonds whose trade freq is low
        train_dictionary.filter_extremes(no_below=no_below, no_above=1., keep_n=2000)
        # num of bonds after filtering
        no_below_num_bonds = len(train_dictionary)

        if no_below_num_bonds <= 5:
            continue

        # filter all traces whose bonds are not in the dictionary
        train_traces = list(filter(lambda x: train_dictionary.doc2idx([x[0]])[0] != -1, train_traces))
        test_traces = list(filter(lambda x: train_dictionary.doc2idx([x[0]])[0] != -1, test_traces))

        # filter this dealer if the transaction count of its filtered traces is small
        if len(train_traces) < 100 or len(test_traces) < 20:
            continue

        new_train_d_dealers[_dealer_index] = train_traces
        new_test_d_dealers[_dealer_index] = test_traces

    utils.write_pkl(cache_path, (new_train_d_dealers, new_test_d_dealers))
    return new_train_d_dealers, new_test_d_dealers


def gen_inputs(_group_file_path, _group_index, _trace_suffix, input_time_steps_list, output_time_steps_list,
               _with_day_off=True, _buy_sell_plan=2, _use_volume=False, save_path='',
               _get_all=False, _get_individual=False, use_cache=True, return_dictionary=False,
               only_buy_y=False, only_sell_y=False):
    """ generate input and ground truth data """

    # load group_dict
    d_dealer_index_2_group_label = utils.load_json(_group_file_path)

    # load d dealers and filter traces and bonds whose freq is low
    train_d_dealers, test_d_dealers = load_d_dealers_after_filtering(_trace_suffix, use_cache=True)

    # d_bond_id_2_freq_type_by_count = utils.load_json(utils.get_relative_dir(
    #     'cache', 'd_bond_id_2_freq_type_by_count.json', root=path.RUNTIME_DIR))
    # d_bond_id_2_freq_type_by_volume = utils.load_json(utils.get_relative_dir(
    #     'cache', 'd_bond_id_2_freq_type_by_volume.json', root=path.RUNTIME_DIR))

    # calculate the number of group members
    if not _get_all and not _get_individual:
        tmp_list = [dealer_index for dealer_index, group_label in d_dealer_index_2_group_label.items()
                    if group_label == _group_index]
        print(f'len_group_member: {len(tmp_list)}')
    elif _get_individual:
        print('len_group_member: 1')
    else:
        print(f'len_group_member: {len(d_dealer_index_2_group_label)}')

    # get total train trace list
    train_trace_list = []
    for dealer_index, traces in train_d_dealers.items():
        if dealer_index not in d_dealer_index_2_group_label or \
                (not _get_all and d_dealer_index_2_group_label[dealer_index] != _group_index):
            continue
        train_trace_list += traces

        # print(dealer_index, len(trace_list), trace_list)

    # sort trace list so that bond traded first could be get smaller bond index (for visualization convenience)
    train_trace_list.sort(key=lambda x: x[-1])

    # get dictionary
    train_doc_list = [list(map(lambda x: x[0], train_trace_list))]
    dictionary = corpora.Dictionary(train_doc_list)
    len_bonds = len(dictionary)
    print(f'total bond num (group {_group_index}): {len_bonds}\n')

    # doc_of_most_freq_bonds = list(filter(lambda x: x[1] == 'most', d_bond_id_2_freq_type_by_count.items()))
    # doc_of_most_freq_bonds = list(map(lambda x: x[0], doc_of_most_freq_bonds))
    #
    # doc_of_more_freq_bonds = list(filter(lambda x: x[1] == 'more', d_bond_id_2_freq_type_by_count.items()))
    # doc_of_more_freq_bonds = list(map(lambda x: x[0], doc_of_more_freq_bonds))
    #
    # doc_of_less_freq_bonds = list(filter(lambda x: x[1] == 'less', d_bond_id_2_freq_type_by_count.items()))
    # doc_of_less_freq_bonds = list(map(lambda x: x[0], doc_of_less_freq_bonds))
    #
    # doc_of_least_freq_bonds = list(filter(lambda x: x[1] == 'least', d_bond_id_2_freq_type_by_count.items()))
    # doc_of_least_freq_bonds = list(map(lambda x: x[0], doc_of_least_freq_bonds))
    #
    # list_token_ids_most_freq_bonds = dictionary.doc2idx(doc_of_most_freq_bonds)
    # list_token_ids_more_freq_bonds = dictionary.doc2idx(doc_of_more_freq_bonds)
    # list_token_ids_less_freq_bonds = dictionary.doc2idx(doc_of_less_freq_bonds)
    # list_token_ids_least_freq_bonds = dictionary.doc2idx(doc_of_least_freq_bonds)
    #
    # list_token_ids_most_freq_bonds = list(filter(lambda x: x >= 0, list_token_ids_most_freq_bonds))
    # list_token_ids_more_freq_bonds = list(filter(lambda x: x >= 0, list_token_ids_more_freq_bonds))
    # list_token_ids_less_freq_bonds = list(filter(lambda x: x >= 0, list_token_ids_less_freq_bonds))
    # list_token_ids_least_freq_bonds = list(filter(lambda x: x >= 0, list_token_ids_least_freq_bonds))
    #
    # list_token_ids_most_freq_bonds.sort()
    # list_token_ids_more_freq_bonds.sort()
    # list_token_ids_less_freq_bonds.sort()
    # list_token_ids_least_freq_bonds.sort()
    #
    # mask_for_most = np.zeros(len(dictionary) + 2, dtype=np.int32)
    # mask_for_more = np.zeros(len(dictionary) + 2, dtype=np.int32)
    # mask_for_less = np.zeros(len(dictionary) + 2, dtype=np.int32)
    # mask_for_least = np.zeros(len(dictionary) + 2, dtype=np.int32)
    #
    # mask_for_most[list_token_ids_most_freq_bonds] = 1
    # mask_for_more[list_token_ids_more_freq_bonds] = 1
    # mask_for_less[list_token_ids_less_freq_bonds] = 1
    # mask_for_least[list_token_ids_least_freq_bonds] = 1
    #
    # print('list_token_ids_most_freq_bonds')
    # print(list_token_ids_most_freq_bonds)
    # print(mask_for_most)
    # print('list_token_ids_more_freq_bonds')
    # print(list_token_ids_more_freq_bonds)
    # print(mask_for_more)
    # print('list_token_ids_less_freq_bonds')
    # print(list_token_ids_less_freq_bonds)
    # print(mask_for_less)
    # print('list_token_ids_least_freq_bonds')
    # print(list_token_ids_least_freq_bonds)
    # print(mask_for_least)
    #
    # _path_mask = utils.get_relative_dir('runtime', 'cache', f'group_{group_index}_mask.pkl')
    # utils.write_pkl(_path_mask, [mask_for_most, mask_for_more, mask_for_less, mask_for_least])
    #
    # print('\ndone')
    #
    # exit()

    if not _get_individual and return_dictionary:
        return dictionary

    dictionary_dict = {}

    # save data to a dict; key would be dealer index, value would be [X, y]
    for dealer_index, train_traces in train_d_dealers.items():
        if dealer_index not in d_dealer_index_2_group_label or \
                (not _get_all and not _get_individual and d_dealer_index_2_group_label[dealer_index] != _group_index):
            continue

        if _get_individual:
            train_doc_list = [list(map(lambda x: x[0], train_traces))]
            dictionary = corpora.Dictionary(train_doc_list)
            len_bonds = len(dictionary)
            print(f'total bond num (dealer {dealer_index}): {len_bonds}\n')

            if return_dictionary:
                dictionary_dict[dealer_index] = dictionary
                continue

        test_traces = test_d_dealers[dealer_index]

        # filter bonds that only appear in test set
        test_traces = list(filter(lambda x: dictionary.doc2idx([x[0]])[0] != -1, test_traces))

        print(f'\ndealer: {dealer_index}, train_traces_num: {len(train_traces)}, test_traces_num: {len(test_traces)}')

        # print('\n------------ filter bonds for a specific freq level ---------------')
        # tmp_train_traces = copy.deepcopy(train_traces)
        # tmp_test_traces = copy.deepcopy(test_traces)
        #
        # tmp_train_traces = list(filter(
        #     lambda x: x[0] in d_bond_id_2_freq_type_by_count and d_bond_id_2_freq_type_by_count[x[0]] == freq_level,
        #     tmp_train_traces))
        # tmp_test_traces = list(filter(
        #     lambda x: x[0] in d_bond_id_2_freq_type_by_count and d_bond_id_2_freq_type_by_count[x[0]] == freq_level,
        #     tmp_test_traces))

        # get train_x, train_y
        train_x, train_y = get_x_y_for_one_dealer(traces=train_traces,
                                                  dictionary=dictionary,
                                                  _input_windows=input_time_steps_list,
                                                  _output_windows=output_time_steps_list,
                                                  pre_traces=None,
                                                  _with_day_off=_with_day_off,
                                                  _buy_sell_plan=_buy_sell_plan,
                                                  _use_volume=_use_volume,
                                                  only_buy_y=only_buy_y,
                                                  only_sell_y=only_sell_y)

        # get test_x, test_y
        test_x, test_y = get_x_y_for_one_dealer(traces=test_traces,
                                                dictionary=dictionary,
                                                _input_windows=input_time_steps_list,
                                                _output_windows=output_time_steps_list,
                                                pre_traces=train_traces,
                                                _with_day_off=_with_day_off,
                                                _buy_sell_plan=_buy_sell_plan,
                                                _use_volume=_use_volume,
                                                only_buy_y=only_buy_y,
                                                only_sell_y=only_sell_y)

        print(
            f'\ttrain_x_shape: ({len(train_x)}, {train_x[0].shape}), train_y_shape: ({len(train_y)}, {len(train_y[0])})')
        print(f'\ttest_x_shape: ({len(test_x)}, {test_x[0].shape}), test_y_shape: ({len(test_y)}, {len(test_y[0])})')

        # create dir for each dealer
        tmp_save_path = save_path
        if _get_individual:
            tmp_save_path = os.path.join(os.path.split(save_path)[0], f'i{dealer_index}')
            if not os.path.exists(tmp_save_path):
                os.mkdir(tmp_save_path)

        utils.write_pkl(os.path.join(tmp_save_path, f'train_{dealer_index}.pkl'), [train_x, train_y])
        utils.write_pkl(os.path.join(tmp_save_path, f'test_{dealer_index}.pkl'), [test_x, test_y])

    if return_dictionary and _get_individual:
        return dictionary_dict


trace_suffix = 'd_dealers_2015_split_by_date_4_types.json'
group_name = 'group_k_means_split_by_date.json'
param_name = 'no_day_off_no_distinguish_buy_sell_use_transaction_count_only_sell'
# param_name = 'no_day_off_distinguish_buy_sell_use_transaction_count'
group_index = 3
get_all = False
get_individual = False
filtering_use_cache = False
only_buy_y = False
only_sell_y = True

input_windows = [5, 10, 15]
output_windows = [2]
buy_sell_plan = 0
with_day_off = False
use_volume = False

if get_all:
    group_index = 'all'

group_file_path = os.path.join(path.ROOT_DIR, 'groups', group_name)

# generate save dir
save_file_path = utils.get_relative_dir('input_data', group_name.split('.')[0], param_name,
                                        # f'group_{group_index}_{freq_level}',
                                        f'group_{group_index}',
                                        root=path.DATA_ROOT_DIR)

gen_inputs(group_file_path, group_index, trace_suffix, input_windows, output_windows,
           with_day_off, buy_sell_plan, use_volume,
           save_path=save_file_path, _get_all=get_all, _get_individual=get_individual, use_cache=filtering_use_cache,
           only_buy_y=only_buy_y, only_sell_y=only_sell_y)
