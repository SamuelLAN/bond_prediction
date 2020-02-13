import os
import shutil
from config import path

# define the ration between training set and test set
# keep_in_src_ratio = 0.88
bound_date = '10-15'
src_dir_name = 'train'
des_dir_name = 'test'

source_dir = path.PREDICTION_DATE_BY_VOLUME_DIR

for _type_name in os.listdir(source_dir):
    src_type_dir = os.path.join(source_dir, _type_name)

    for _year in os.listdir(src_type_dir):
        src_year_dir = os.path.join(src_type_dir, _year)
        tmp_bound_date = f'doc_{_year}-{bound_date}'

        for _volume in os.listdir(src_year_dir):
            src_volume_dir = os.path.join(src_year_dir, _volume)

            print(f'\nStart moving dateset {_type_name}/{_year}/{_volume} ...')

            object_list = os.listdir(src_volume_dir)
            len_obj_list = len(object_list)

            for i, object_name in enumerate(object_list):
                progress = float(i + 1) / len_obj_list * 100.
                print('\rprogress: %.2f%% ' % progress, end='')

                object_dir = os.path.join(src_volume_dir, object_name)
                src_object_dir = os.path.join(object_dir, src_dir_name)
                des_object_dir = os.path.join(object_dir, des_dir_name)

                src_date_list = os.listdir(src_object_dir)
                des_date_list = os.listdir(des_object_dir)

                len_src_dates = len(src_date_list)
                len_des_dates = len(des_date_list)

                if des_date_list[0] == tmp_bound_date:
                    continue

                move_date_list = []

                if des_date_list[0] < tmp_bound_date:
                    index = -1
                    for i, v in enumerate(des_date_list[:-1]):
                        if v <= tmp_bound_date < des_date_list[i + 1]:
                            index = i
                            break

                    if index == -1:
                        continue

                    move_date_list = des_date_list[:index]
                    for file_name in move_date_list:
                        src_file_path = os.path.join(src_object_dir, file_name)
                        des_file_path = os.path.join(des_object_dir, file_name)
                        shutil.move(des_file_path, src_file_path)

                elif src_date_list[-1] > tmp_bound_date:
                    index = len_src_dates - 1
                    while index >= 1:
                        if src_date_list[index - 1] < tmp_bound_date <= src_date_list[index]:
                            break
                        index -= 1

                    if index < 5:
                        continue

                    move_date_list = src_date_list[index:]
                    for file_name in move_date_list:
                        src_file_path = os.path.join(src_object_dir, file_name)
                        des_file_path = os.path.join(des_object_dir, file_name)
                        shutil.move(src_file_path, des_file_path)

                # train_end_idx = int(len_src_dates * keep_in_src_ratio)
                # move_date_list = date_list[train_end_idx:]
                #
                # for file_name in move_date_list:
                #     src_file_path = os.path.join(src_object_dir, file_name)
                #     des_file_path = os.path.join(des_object_dir, file_name)
                #     shutil.move(src_file_path, des_file_path)

            print('\nFinish moving ')

print('\ndone')
