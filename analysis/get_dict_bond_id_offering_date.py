import os
import numpy as np
from config import path
from lib import utils

path_d_issue_id_offering_date = os.path.join(path.DATA_ROOT_DIR, 'dict_issue_id_offering_date.json')
dict_issue_id_offering_date = utils.load_json(path_d_issue_id_offering_date)

path_pkl_2015 = os.path.join(path.TRACE_DIR, 'finra_trace_2015.pkl')
data = utils.load_pkl(path_pkl_2015)

print('\nstart converting data ...')
data = np.array(data)
data = list(map(lambda x: {'bond_id': x[0], 'issue_id': x[16]}, data))
print('finish converting ')

dict_skip_bond = {}

dict_bond_id_offering_date = {}
for v in data:
    bond_id = v['bond_id']
    issue_id = str(int(v['issue_id']))

    if issue_id not in dict_issue_id_offering_date:
        dict_skip_bond[bond_id] = True
        print('------------------------')
        print(bond_id, issue_id)
        continue

    offering_date = dict_issue_id_offering_date[issue_id]
    dict_bond_id_offering_date[bond_id] = offering_date

save_path = os.path.join(path.DATA_ROOT_DIR, 'dict_bond_id_offering_date.json')
utils.write_json(save_path, dict_bond_id_offering_date)

print('done')
print(f'skip bonds: {len(dict_skip_bond)}')
