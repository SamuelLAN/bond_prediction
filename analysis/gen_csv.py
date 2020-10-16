import os
import pandas as pd
import numpy as np

cur_dir = os.path.split(os.path.abspath(__file__))[0]
root_dir = os.path.split(cur_dir)[0]

result_csv_path = os.path.join(root_dir, 'runtime', 'experiments.csv')

data = pd.read_csv(result_csv_path)

# ----------------------------------------

columns = list(data.columns)
data = np.array(data.iloc[:])

for i, v in enumerate(columns):
    print(i, v)

experiment_result_dict = {}
data_result_dict = {}
individual_exp_dict = {}

for i, val in enumerate(data):
    model_name = val[0]
    group_name = val[1]
    group_file_name = val[3]
    
    if 'Spectral_Clustering' in group_name:
        if '_with_model_input_features' in group_name:
            group_name = 'SC_trace'
        else:
            group_name = 'SC'
    else:
        group_name = 'K-means'

    group_index = group_file_name.split('_')[1]
    test_f1 = val[19]

    if group_name == 'SC_trace':
        voc_size = val[8]
        # train_size =
        train_label_cardinality = val[4]
        train_label_density = val[6]
        individual_exp_dict[group_index] = [group_index, voc_size, train_label_cardinality, train_label_density, test_f1]

        if group_index[0] == 'i':
            continue

    data_key = (group_name, group_index)
    if data_key not in data_result_dict:
        data_result_dict[data_key] = val[4:12]
        
    if model_name not in experiment_result_dict:
        experiment_result_dict[model_name] = {}
        
    if data_key not in experiment_result_dict[model_name]:
        experiment_result_dict[model_name][data_key] = test_f1

data_csv = 'group,' \
           'train_label_cardinality,test_label_cardinality,' \
           'train_label_density,test_label_density,' \
           'voc_size,max_time_window,train_size,test_size\n'

data_keys = list(data_result_dict.items())
data_keys.sort(key=lambda x: (x[0][0], x[1][0]))

data_keys = list(map(lambda x: x[0], data_keys))
# data_keys = list(map(lambda x: x[0], data_result_dict.items()))
# data_keys.sort()

groups = list(map(lambda x: x[0] + ':' + x[1], data_keys))
experiments_csv = 'model_name,' + ','.join(groups) + ',SC_trace_avg\n'

for i, val in enumerate(data_keys):
    data_csv += groups[i] + ',' + ','.join(list(map(str, data_result_dict[val]))) + '\n'

models = list(map(lambda x: x[0], experiment_result_dict.items()))
models.sort()

individuals = list(map(lambda x: x[1], individual_exp_dict.items()))
individuals.sort(key=lambda x: [x[2], x[-1]])

individual_csv = 'group_index,voc_size,label_cardinality,label_density,test_f1\n'
individual_csv += '\n'.join(list(map(lambda x: ','.join(list(map(str, x))), individuals)))

for model_name in models:
    tmp_exp_str = model_name
    SC_trace_avg = 0.
    c = 0.
    for i, val in enumerate(data_keys):
        tmp_exp_str += ','
        if val in experiment_result_dict[model_name]:
            score = experiment_result_dict[model_name][val]
            tmp_exp_str += str(score)

            if val[0] == 'SC_trace':
                SC_trace_avg += float(score)
                c += 1
        else:
            tmp_exp_str += '-'
    SC_trace_avg = SC_trace_avg / c
    experiments_csv += tmp_exp_str + f',{SC_trace_avg}\n'


with open(os.path.join(root_dir, 'runtime', 'groups.csv'), 'wb') as f:
    f.write(data_csv.encode('utf-8'))

with open(os.path.join(root_dir, 'runtime', 'experiments_of_test_f1.csv'), 'wb') as f:
    f.write(experiments_csv.encode('utf-8'))

with open(os.path.join(root_dir, 'runtime', 'experiments_of_test_f1_for_individuals.csv'), 'wb') as f:
    f.write(individual_csv.encode('utf-8'))

# print(data.columns)
# print(data)


