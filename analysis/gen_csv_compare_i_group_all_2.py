import os
import json
import numpy as np
from lib import utils
from config import path

cur_dir = os.path.abspath(os.path.split(__file__)[0])
root_dir = os.path.split(cur_dir)[0]

# choose_group = 'group_spectral_clustering_with_patterns_info_cluster_6_split_by_date'
choose_group = 'group_k_means_split_by_date'
# choose_group = 'group_k_means_cluster_3_split_by_date'
model_individual_log_path = r'D:\Github\bond_prediction_branches\dev_2.0\bond_prediction\runtime\model_individual.log'
model_log_path = r'D:\Github\bond_prediction_branches\dev_2.0\bond_prediction\runtime\model_v4.log'
model_log_path_v1 = r'D:\Github\bond_prediction_branches\dev_2.0\bond_prediction\runtime\model.log'
model_log_path_v2 = r'D:\Github\bond_prediction_branches\dev_2.0\bond_prediction\runtime\model_v5.log'
group_json_path = os.path.join(root_dir, 'groups', f'{choose_group}.json')

group_json = utils.load_json(group_json_path)
group_index_set = list(set(list(group_json.values())))


# d_dealer_suffix = r'_d_dealers_2015_split_by_date.json'
# train_d_dealers = os.path.join(path.D_DEALERS_TRACE_DIR, 'train' + d_dealer_suffix)
# test_d_dealers = os.path.join(path.D_DEALERS_TRACE_DIR, 'test' + d_dealer_suffix)
#
# train_d_dealers = utils.load_json(train_d_dealers)
# test_d_dealers = utils.load_json(test_d_dealers)


# # read model.log
# with open(model_individual_log_path, 'rb') as f:
#     content_individual = f.read().decode('utf-8').strip().split('\n\n\n')
#
# with open(model_log_path, 'rb') as f:
#     content = f.read().decode('utf-8').strip().split('\n\n\n')


def convert(x):
    if x and x[0] == '{' and x[-1] == '}':
        x = x.replace("'", '"').replace('False', '"False"').replace('True', '"True"')
        return json.loads(x)
    return x


def parse_one_model_log(content):
    data = {}
    content = content.replace('\r', '').split('\n')
    content = list(filter(lambda x: ':' in x, content))
    content = list(map(lambda x: [x[:x.index(':')].strip(), x[x.index(':') + 1:].strip()], content))

    is_individual_result = False
    for k, v in content:
        if k == 'dealer_results':
            is_individual_result = True
            data[k] = []
            continue

        if not is_individual_result:
            data[k] = convert(v)
        else:
            tmp_result = convert(v)
            tmp_dealer_index = k.replace('.pkl', '')
            tmp_i_group_index = group_json[tmp_dealer_index]
            data['dealer_results'].append(
                np.array([tmp_dealer_index, tmp_i_group_index,
                          [tmp_result['precision'], tmp_result['recall'], tmp_result['f1']]]))

    result = data['test_result_dict']

    data['is_individual'] = False if data['dealer_results'] else True
    data['is_all'] = True if not data['is_individual'] and data['data_param']['group_file_name'] == 'all' else False
    data['dealer_index'] = '' if data['dealer_results'] else data['data_param']['group_file_name'][1:]
    data['group_index'] = data['data_param']['group_file_name'].split('_')[1] \
        if not data['is_individual'] and not data['is_all'] else ''
    data['i_group_index'] = group_json[data['dealer_index']] if data['is_individual'] else ''
    data['p_r_f1'] = np.array([result['precision'], result['recall'], result['f1']])
    data['group_path'] = os.path.join(root_dir, 'groups', data['data_param']['group_name'] + '.json')

    return data


def parse_model_log(_path, total_dict, group_name):
    with open(_path, 'rb') as f:
        content_list = f.read().decode('utf-8').strip().split('\n\n\n')

    for content in content_list:
        tmp_data = parse_one_model_log(content)
        tmp_group_name = tmp_data['data_param']['group_name']
        if tmp_group_name != group_name:
            continue

        _model_name = tmp_data['model_name']
        if _model_name not in total_dict:
            total_dict[_model_name] = {
                'individual_results': [],
                'group_results': [],
                'all_results': [],
                'p_r_f1': {},
            }

        is_individual = tmp_data['is_individual']
        is_all = tmp_data['is_all']
        dealer_index = tmp_data['dealer_index']
        group_index = tmp_data['group_index']
        i_group_index = tmp_data['i_group_index']
        p_r_f1 = tmp_data['p_r_f1']
        dealer_results = tmp_data['dealer_results']

        if is_individual:
            total_dict[_model_name]['individual_results'].append([dealer_index, i_group_index, p_r_f1])

        elif is_all:
            total_dict[_model_name]['all_results'] = dealer_results
            total_dict[_model_name]['p_r_f1']['all'] = p_r_f1

        else:
            total_dict[_model_name]['group_results'] += dealer_results
            total_dict[_model_name]['p_r_f1'][str(group_index)] = p_r_f1


summary = {}
parse_model_log(model_individual_log_path, summary, choose_group)
parse_model_log(model_log_path, summary, choose_group)

parse_model_log(model_log_path_v1, summary, choose_group)
parse_model_log(model_log_path_v2, summary, choose_group)


# parse_model_log(model_log_path_v2, summary, choose_group)


def calculate_group_from_individual(results, suffix, _dict):
    if not results:
        return

    tmp_results = np.array(list(map(lambda x: x[2], results)))
    _dict['p_r_f1'][f'{suffix}_avg'] = np.mean(tmp_results, axis=0)
    _dict['p_r_f1'][f'{suffix}_std'] = np.std(tmp_results, axis=0)

    for group_index in group_index_set:
        tmp_results = list(filter(lambda x: int(x[1]) == group_index, results))
        if tmp_results:
            tmp_results = np.array(list(map(lambda x: x[2], tmp_results)))
            tmp_results = list(filter(lambda x: x[-1], tmp_results))
            _dict['p_r_f1'][f'{suffix}_g{group_index}'] = np.mean(tmp_results, axis=0)
            _dict['p_r_f1'][f'{suffix}_g{group_index}_std'] = np.std(tmp_results, axis=0)


def summary_info(total_dict):
    for _model_name, val in total_dict.items():
        calculate_group_from_individual(val['individual_results'], 'i', val)
        calculate_group_from_individual(val['group_results'], 'g', val)
        calculate_group_from_individual(val['all_results'], 'a', val)

        print('\n-----------------------')

    return total_dict


def round_f(x):
    return str(round(float(x) * 100, 2))


def convert_2_csv(total_dict, _path):
    # results_order = ['3', '0', '1', '2', '4', '5', '6', '7',
    #                  'g_g3', 'g_g0', 'g_g1', 'g_g2', 'g_g4', 'g_g5', 'g_g6', 'g_g7', 'g_avg',
    #                  'i_g3', 'i_g0', 'i_g1', 'i_g2', 'i_avg',
    #                  'a_g3', 'a_g0', 'a_g1', 'a_g2', 'a_avg']

    results_order = [
        '3', '0', '1', '2',
        'g_g3', 'g_g0', 'g_g1', 'g_g2', 'g_avg',
        'i_g3', 'i_g0', 'i_g1', 'i_g2', 'i_avg',
        'a_g3', 'a_g0', 'a_g1', 'a_g2', 'a_avg'
    ]

    # headline = ['least', 'less', 'more', 'most', 'l_4', 'l_5', 'l_6', 'l_7',
    headline = ['least', 'less', 'more', 'most',
                # 'g_least', 'g_less', 'g_more', 'g_most', 'g_g4', 'g_g5', 'g_g6', 'g_g7', 'g_avg',
                'g_least', 'g_less', 'g_more', 'g_most', 'g_avg',
                # 'g_g3', 'g_g0', 'g_g1', 'g_g2', 'g_g4', 'g_g5', 'g_g6', 'g_g7', 'g_avg',
                'i_least', 'i_less', 'i_more', 'i_most', 'i_avg',
                'a_least', 'a_less', 'a_more', 'a_most', 'a_avg']
    headline = list(map(lambda x: x + ' P', headline)) + \
               list(map(lambda x: x + ' R', headline)) + \
               list(map(lambda x: x + ' F1', headline))
    headline = ['model_name'] + headline

    def __get_val(_k, _val, _index):
        _tmp_p_r_f1 = _val['p_r_f1']
        if _k not in _tmp_p_r_f1:
            return ''

        std_k = f"{_k.replace('_avg', '')}_std"
        if std_k not in _tmp_p_r_f1:
            return round_f(_tmp_p_r_f1[_k][_index])
        else:
            return f'{round_f(_tmp_p_r_f1[_k][_index])}/{round_f(_tmp_p_r_f1[std_k][_index])}'

    data = [headline]
    for _model_name, val in total_dict.items():
        # tmp_result = [round_f(val['p_r_f1'][k][0]) if k in val['p_r_f1'] else '' for k in results_order] + \
        #              [round_f(val['p_r_f1'][k][1]) if k in val['p_r_f1'] else '' for k in results_order] + \
        #              [round_f(val['p_r_f1'][k][2]) if k in val['p_r_f1'] else '' for k in results_order]

        tmp_result = [__get_val(k, val, 0) for k in results_order] + \
                     [__get_val(k, val, 1) for k in results_order] + \
                     [__get_val(k, val, 2) for k in results_order]

        # tmp_result = [round_f(val['p_r_f1'][k][2]) if k in val['p_r_f1'] else '' for k in results_order]

        data.append([_model_name] + tmp_result)

    csv_data = list(map(lambda x: ','.join(list(map(str, x))), data))
    csv_data = '\n'.join(csv_data)

    with open(_path, 'wb') as f:
        f.write(csv_data.encode('utf-8'))

    ms = ['fc_with_concatenation_of_feature_vectors',
          'lstm_with_feature_vectors',
          'bilstm_with_feature_vectors',
          'transformer_with_summed_bond_embeddings',
          'transformer_rezero_with_summed_bond_embeddings',
          'transformer_modified_zero_with_summed_bond_embeddings']

    # latex_data = list(filter(lambda x: x[0] in ms, data))
    # latex_data = list(map(lambda x: ' & '.join(list(map(str, x))), data))
    # latex_data = '\n'.join(latex_data)

    results_order = ['g_g0', 'g_g1', 'g_g2']

    print('\n------------------------------------------')

    for _model_name, val in total_dict.items():
        # if _model_name not in ms:
        #     continue

        # l = [_model_name,] + [' & '.join(list(map(round_f, val['p_r_f1'][k]))) if k in val['p_r_f1'] else '- & - & -' for k in
        #          results_order]

        l = [_model_name, ] + [round_f(val['p_r_f1'][k][2]) if k in val['p_r_f1'] else '-' for k in results_order]
        l = ' & '.join(l)
        print(l)

    results_order = [
        ['most', 'i_g2', 'g_g2', 'a_g2'],
        ['more', 'i_g1', 'g_g1', 'a_g1'],
        ['less', 'i_g0', 'g_g0', 'a_g0'],
        ['least', 'i_g3', 'g_g3', 'a_g3'],
        ['avg', 'i_avg', 'g_avg', 'a_avg'],
    ]

    print('\n------------------------------------------')
    for _model_name, val in total_dict.items():
        if _model_name != 'transformer_modified_zero_with_summed_bond_embeddings':
            continue

        for _order in results_order:
            l = _order[:1] + \
                [' & '.join(list(map(round_f, val['p_r_f1'][k]))) if k in val['p_r_f1'] else '- & - & -' for k in
                 _order[1:]]

            l = ' & '.join(l)
            print(l)


# transformer_modified_zero_with_summed_bond_embeddings

summary = summary_info(summary)

convert_2_csv(summary, os.path.join(root_dir, 'runtime', 'exp_results_new11.csv'))

exit()

for model_name, _dict in summary.items():
    print('\n--------------------------------------')
    print(model_name)
    for k, v in _dict.items():
        print(f'{k}:')
        if isinstance(v, list):
            if len(v) > 20:
                continue
            for vv in v:
                print(f'\t{vv}')
        if isinstance(v, dict):
            if len(v) > 20:
                continue
            for kk, vv in v.items():
                print(f'\t{kk}: {vv}')

"""

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

# for group_index, result_dict in individual_result.items():
#     group_index = 'transformer2_' + group_index
#     for group_file_name, result in result_dict.items():
#         dealer_index = group_file_name.split('.')[0]
# 
#         if dealer_index not in data:
#             data[dealer_index] = {}
#         data[dealer_index][group_index] = result['f1']

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
"""
