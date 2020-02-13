# only use once
import os
import json
from config import path

source_dir = path.BONDS_BY_DEALER_DATE_DOC_DIR
dealer_indices = os.listdir(source_dir)
len_indices = len(dealer_indices)

for i, dealer_index in enumerate(dealer_indices):
    if i % 2 == 0:
        progress = float(i + 1) / len_indices * 100.
        print('\rprogress: %.2f%% ' % progress, end='')

    sub_source_dir = os.path.join(source_dir, dealer_index)
    for file_name in os.listdir(sub_source_dir):
        file_path = os.path.join(sub_source_dir, file_name)

        with open(file_path, 'rb') as f:
            data = json.load(f)

        if not data:
            os.remove(file_path)
            # print(file_path)

print('\ndone')
