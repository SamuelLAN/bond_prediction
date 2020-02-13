from config import path
from lib.utils import load_json, write_json

l = load_json(path.TOPIC_BONDS_LIST_JSON)

topic_bonds = set()
threshold = 0.3

for bonds in l:
    percentage = 0.
    for i, v in enumerate(bonds):
        bond_id = v[0]
        weight = v[1]

        if weight < 0.1:
            break

        percentage += weight
        topic_bonds.add(bond_id)

        if percentage >= 0.5:
            break

topic_bonds = list(topic_bonds)
topic_bonds.sort()

write_json(path.TOPIC_BONDS_JSON, topic_bonds)

print(len(topic_bonds))
print(topic_bonds)
print('done')
