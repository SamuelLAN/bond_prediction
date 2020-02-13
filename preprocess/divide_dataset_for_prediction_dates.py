import os
import shutil
from config import path
from lib.utils import load_json

# define the ration between training set and test set
train_ratio = 0.8
test_ratio = 0.2


def __check_one_volume(_dir_path):
    count = 0
    for _file_name in os.listdir(_dir_path):
        _data = load_json(os.path.join(_dir_path, _file_name))
        count += len(_data)
    return count


source_dir = path.PREDICTION_DATE_DIR
destination_dir = path.PREDICTION_DATE_BY_VOLUME_DIR

range_dict = {
    '100_1k': [100, 1000],
    '1k_10k': [1000, 10000],
    '10k_100k': [10000, 100000],
    '100k_1m': [100000, 1000000],
    '1m_10m': [1000000, 10000000],
}

for _type_name in os.listdir(source_dir):
    src_type_dir = os.path.join(source_dir, _type_name)
    des_type_dir = os.path.join(destination_dir, _type_name)

    if not os.path.isdir(des_type_dir):
        os.mkdir(des_type_dir)

    for _year in os.listdir(src_type_dir):
        src_year_dir = os.path.join(src_type_dir, _year)
        des_year_dir = os.path.join(des_type_dir, _year)

        if not os.path.isdir(des_year_dir):
            os.mkdir(des_year_dir)

        for k, v in range_dict.items():
            des_volume_dir = os.path.join(des_year_dir, k)
            if not os.path.isdir(des_volume_dir):
                os.mkdir(des_volume_dir)

        print(f'\nStart dividing dateset from {src_year_dir} to {des_year_dir} ...')

        object_list = os.listdir(src_year_dir)
        len_obj_list = len(object_list)

        for i, object_name in enumerate(object_list):
            progress = float(i + 1) / len_obj_list * 100.
            print('\rprogress: %.2f%% ' % progress, end='')

            src_object_dir = os.path.join(src_year_dir, object_name)
            volume = __check_one_volume(src_object_dir)

            volume_level = ''
            for k, v in range_dict.items():
                if v[0] <= volume < v[1]:
                    volume_level = k
                    break

            if not volume_level:
                continue

            des_object_dir = os.path.join(des_year_dir, volume_level, object_name)
            if not os.path.isdir(des_object_dir):
                os.mkdir(des_object_dir)

            des_object_train_dir = os.path.join(des_object_dir, 'train')
            des_object_test_dir = os.path.join(des_object_dir, 'test')

            if not os.path.isdir(des_object_train_dir):
                os.mkdir(des_object_train_dir)
            if not os.path.isdir(des_object_test_dir):
                os.mkdir(des_object_test_dir)

            date_list = os.listdir(src_object_dir)
            len_dates = len(date_list)
            train_end_idx = int(len_dates * train_ratio)

            train_date_list = date_list[:train_end_idx]
            test_date_list = date_list[train_end_idx:]

            for file_name in train_date_list:
                src_file_path = os.path.join(src_object_dir, file_name)
                des_file_path = os.path.join(des_object_train_dir, file_name)
                shutil.copy2(src_file_path, des_file_path)

            for file_name in test_date_list:
                src_file_path = os.path.join(src_object_dir, file_name)
                des_file_path = os.path.join(des_object_test_dir, file_name)
                shutil.copy2(src_file_path, des_file_path)

        print('\nFinish dividing ')

print('\ndone')
