import os
import json
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from config import path
from preprocess.filter_dealer_list import top_n, by_range

# define the ration between training set and test set
train_ratio = 0.9
test_ratio = 0.1

# define directories of training set and test set
source_dir = path.BONDS_BY_DEALER_DATE_DOC_DIR

# dealer_indices = os.listdir(source_dir)

dealer_1k_10k_indices = by_range(1000, 10000)
dealer_10k_100k_indices = by_range(10000, 100000)
dealer_100k_1m_indices = by_range(100000, 1000000)
dealer_1m_10m_indices = by_range(1000000, 10000000)
dealer_10m_indices = by_range(10000000)

random.shuffle(dealer_1k_10k_indices)
random.shuffle(dealer_10k_100k_indices)
random.shuffle(dealer_100k_1m_indices)
random.shuffle(dealer_1m_10m_indices)
random.shuffle(dealer_10m_indices)


def get_train_test_dates(dates):
    dates.sort()
    len_dates = len(dates)
    train_index_end = int(train_ratio * len_dates)
    train_dates = dates[: train_index_end]
    test_dates = dates[train_index_end:]
    return train_dates, test_dates


def cmd(command):
    return os.popen(command).read()


def move_files(_dealer_indices, destination_dir):
    print(f'\nStart copying files from {source_dir} to {destination_dir} ...')

    length = len(_dealer_indices)
    for i, dealer_index in enumerate(_dealer_indices):
        if i % 2 == 0:
            progress = float(i + 1) / length * 100.
            print('\rprogress: %.2f%% ' % progress, end='')

        source_path = os.path.join(source_dir, dealer_index)

        destination_path = os.path.join(destination_dir, dealer_index)
        train_destination_dir = os.path.join(destination_path, 'train')
        test_destination_dir = os.path.join(destination_path, 'test')
        path.mk_if_not_exist([destination_path, train_destination_dir, test_destination_dir])

        dates = os.listdir(source_path)
        train_dates, test_dates = get_train_test_dates(dates)

        for _date in train_dates:
            src_file_path = os.path.join(source_path, _date)
            des_file_path = os.path.join(train_destination_dir, _date)
            shutil.copy2(src_file_path, des_file_path)

        for _date in test_dates:
            src_file_path = os.path.join(source_path, _date)
            des_file_path = os.path.join(test_destination_dir, _date)
            shutil.copy2(src_file_path, des_file_path)

    print(f'\nFinish copying ')


move_files(dealer_1k_10k_indices, path.ORIGIN_VOLUME_1k_10k_DIR)
move_files(dealer_10k_100k_indices, path.ORIGIN_VOLUME_10k_100k_DIR)
move_files(dealer_100k_1m_indices, path.ORIGIN_VOLUME_100k_1m_DIR)
move_files(dealer_1m_10m_indices, path.ORIGIN_VOLUME_1m_10m_DIR)
move_files(dealer_10m_indices, path.ORIGIN_VOLUME_10m_DIR)

print('\ndone')
