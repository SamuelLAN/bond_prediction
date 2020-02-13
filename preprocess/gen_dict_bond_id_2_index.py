import os
import json
from config import path


d_bond = {}

for file_name in os.listdir(path.BONDS_BY_DEALER_DIR):

    # load bonds data
    file_path = os.path.join(path.BONDS_BY_DEALER_DIR, file_name)
    with open(file_path, 'r') as f:
        bonds = json.load(f)

    for i, v in enumerate(bonds):
        if isinstance(v[0], float):
            continue

        d_bond[v[0]] = True

l_bond = list(d_bond.items())
l_bond.sort(key=lambda x: x[0])
l_bond = list(map(lambda x: x[0], l_bond))

d_bond_id_2_index = {}
d_bond_index_2_id = {}

for bond_index, bond_id in enumerate(l_bond):
    d_bond_id_2_index[bond_id] = bond_index
    d_bond_index_2_id[bond_index] = bond_id

with open(path.DICT_BOND_ID_2_INDEX_JSON, 'w') as f:
    json.dump(d_bond_id_2_index, f)

with open(path.DICT_BOND_INDEX_2_ID_JSON, 'w') as f:
    json.dump(d_bond_index_2_id, f)

print(len(d_bond_id_2_index))
print('done')
