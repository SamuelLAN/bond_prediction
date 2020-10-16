import os
import shutil
from lib import utils

save_best_model_num = 1

model_dir = utils.get_relative_dir('runtime', 'models')
tb_dir = utils.get_relative_dir('runtime', 'tensorboard')

for model_name in os.listdir(model_dir):
    tmp_model_dir = os.path.join(model_dir, model_name)
    print(f'\nchecking {tmp_model_dir} ...')

    for _date in os.listdir(tmp_model_dir):
        date_dir = os.path.join(tmp_model_dir, _date)
        model_list = os.listdir(date_dir)

        print(f'\tchecking {_date}')

        # if model dir is empty, delete the model dir and its tensorboard files
        if not model_list:
            tmp_date_tb_dir = os.path.join(tb_dir, model_name, _date)
            shutil.rmtree(tmp_date_tb_dir)
            os.removedirs(date_dir)
            print(f'\tremove {_date}')

        # if model dir is not empty, only save the best models
        else:
            model_list.sort(reverse=True)
            for model_file_name in model_list[1:]:
                model_path = os.path.join(date_dir, model_file_name)
                os.remove(model_path)
                print(f'\t\tremove {model_file_name}')

print('\ndone')
