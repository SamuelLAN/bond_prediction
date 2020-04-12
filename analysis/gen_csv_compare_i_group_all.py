import os
import json
import numpy as np
from lib import utils
from config import path

individual_result_path = r'D:\Github\bond_prediction_branches\dev_2.0\bond_prediction\runtime\individual_result.json'
model_individual_log_path = r'D:\Github\bond_prediction_branches\dev_2.0\bond_prediction\runtime\model_v2.log'
model_log_path = r'D:\Github\bond_prediction_branches\dev_2.0\bond_prediction\runtime\model.log'

d_dealer_suffix = r'_d_dealers_2015_split_by_date.json'
train_d_dealers = os.path.join(path.D_DEALERS_TRACE_DIR, 'train' + d_dealer_suffix)
test_d_dealers = os.path.join(path.D_DEALERS_TRACE_DIR, 'test' + d_dealer_suffix)

train_d_dealers = utils.load_json(train_d_dealers)
test_d_dealers = utils.load_json(test_d_dealers)

individual_result = utils.load_json(individual_result_path)

# read model.log
with open(model_individual_log_path, 'rb') as f:
    content_individual = f.read().decode('utf-8').strip().split('\n\n\n')

with open(model_log_path, 'rb') as f:
    content = f.read().decode('utf-8').strip().split('\n\n\n')


data = {}

# parse model_log
for i, lines in enumerate(content):
    # parse data line by line
    lines = lines.strip().split('\n')

    model_name = list(filter(lambda x: x[:len('model_name')] == 'model_name', lines))[0].split(':')[1].strip()
    _dealer_results = lines[11:]
    _dealer_results = list(map(lambda x: x.strip().split(': {'), _dealer_results))
    _data_params = json.loads(lines[8].strip('data_param: ').replace("'", '"'))
    group_name = _data_params['group_name']
    group_file_name = _data_params['group_file_name']
    if 'spectral_clustering_with_patterns_info' in group_name:
        if 'cluster_6' in group_name:
            group_file_name = 'sc_trace_cluster_6_' + group_file_name
        else:
            group_file_name = 'sc_trace_' + group_file_name
    elif 'k_means' in group_name:
        if 'cluster_6' in group_name:
            group_file_name = 'k_means_cluster_6_' + group_file_name
        else:
            group_file_name = 'k_means_' + group_file_name

    group_index = model_name + '_' + group_file_name
    _results = list(map(lambda x: [x[0].strip('.pkl'), json.loads('{' + x[1].replace("'", '"'))], _dealer_results))

    for dealer_index, _result in _results:
        if dealer_index not in data:
            data[dealer_index] = {}
        data[dealer_index][group_index] = _result['f1']


for i, lines in enumerate(content_individual):
    # parse data line by line
    lines = lines.strip().split('\n')
    lines = list(filter(lambda x: x[:len('test_result_dict')] == 'test_result_dict' or
                                  x[:len('data_param')] == 'data_param', lines))

    # get result and dealer index
    result = json.loads(lines[0][len('test_result_dict: '):].strip().replace("'", '"'))
    dealer_index = str(json.loads(lines[1][len('data_param: '):].strip())['group_file_name'][1:])

    # add data record to data dict
    if dealer_index not in data:
        data[dealer_index] = {}
    data[dealer_index]['transformer2_individual'] = result['f1']

for group_index, result_dict in individual_result.items():
    group_index = 'transformer2_' + group_index
    for group_file_name, result in result_dict.items():
        dealer_index = group_file_name.split('.')[0]

        if dealer_index not in data:
            data[dealer_index] = {}
        data[dealer_index][group_index] = result['f1']

# for dealer_index, f1 in data.items():
#     print(dealer_index, f1)

new_data_first = ['dealer_index', 'transaction_count', 'transformer2_all', 'transformer2_individual', 'transformer2_k_means', 'transformer2_sc_trace_cluster_6', 'transformer2_sc_trace', 'k_means_group_index', 'sc_trace_cluster_6_group_index', 'sc_trace_group_index']
new_data = []
for dealer_index, f1_dict in data.items():
    if 'transformer2_individual' not in f1_dict:
        f1_dict['transformer2_individual'] = ''

    # get transaction count
    transaction_count = len(train_d_dealers[dealer_index]) if dealer_index in train_d_dealers \
        else len(test_d_dealers[dealer_index])

    # get group index
    keys = list(f1_dict.keys())
    keys.remove('transformer2_all')
    keys.remove('transformer2_individual')
    keys.sort()
    group_indexs = keys

    # sort results
    old_f1_list = list(f1_dict.items())
    old_f1_list.sort(key=lambda x: x[0])

    f1_list = list(map(lambda x: x[1], old_f1_list))

    # integrate results
    tmp_list = [dealer_index, transaction_count] + f1_list + group_indexs
    new_data.append(tmp_list)

new_data.sort(key=lambda x: x[1], reverse=True)
avg_result_group = np.mean(np.array(list(map(lambda x: x[1:-3], new_data))), axis=0)
# avg_result_individual = np.mean(np.array(list(filter(lambda a: a, list(map(lambda x: x[-3], new_data))))), axis=0)
new_data.append(['average'] + list(avg_result_group) + ['k_means', 'sc_trace_cluster_6', 'sc_trace'])
new_data.insert(0, new_data_first)

for val in new_data:
    print(val)


def convert(x):
    if isinstance(x, float):
        return str(round(x, 4))
    return str(x)


new_data = '\n'.join(list(map(lambda x: ','.join(list(map(convert, x))), new_data)))
with open(utils.get_relative_dir('runtime', 'exp_individual.csv'), 'wb') as f:
    f.write(new_data.encode('utf-8'))

# data = list(map(lambda x: [x[0]] + x[1].values(), data.items()))
