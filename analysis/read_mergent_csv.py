import os
from config import path
from lib.utils import decode_2_utf8, write_json

file_path = os.path.join(path.DATA_ROOT_DIR, 'Mergent_FISD_Bonds_Issues_full.csv')
with open(file_path, 'rb') as f:
    content = f.readlines()

data = list(map(decode_2_utf8, content))
data = list(map(lambda x: x.strip().split(','), data))

columns = data[0]
data = data[1:]

# for i, v in enumerate(columns):
#     print(i, v, data[0][i], data[1][i])

index_issue_id = 0
index_offering_date = 32

dict_issue_id_offering_date = {}

for v in data:
    issue_id = v[index_issue_id]
    offering_date = v[index_offering_date]

    if issue_id in dict_issue_id_offering_date:
        print('------------------------')
        print(issue_id, offering_date)

    offering_date = f'{offering_date[:4]}-{offering_date[4:6]}-{offering_date[-2:]}'
    dict_issue_id_offering_date[issue_id] = offering_date

save_path = os.path.join(path.DATA_ROOT_DIR, 'dict_issue_id_offering_date.json')
write_json(save_path, dict_issue_id_offering_date)

print('done')
