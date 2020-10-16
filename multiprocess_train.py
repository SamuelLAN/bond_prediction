import os
import time
from multiprocessing import Process
from train_for_individual import Train


def train(_dealer_index):
    from config.load import LOG, group_path
    LOG['group_file_name'] = _dealer_index
    Train.MODEL_DIR = time.strftime('%Y_%m_%d_%H_%M_%S')

    tmp_group_path = os.path.join(os.path.split(group_path)[0], _dealer_index)
    o_train = Train(_group_path=tmp_group_path, use_generator=False)
    o_train.run()


if __name__ == '__main__':
    from config.load import group_path
    group_dir = os.path.split(group_path)[0]
    dir_list = list(filter(lambda x: x[0].lower() == 'i', os.listdir(group_dir)))
    len_dir = len(dir_list)

    for i, dealer_index in enumerate(dir_list):
        # if dealer_index < 'i1302':
        #     continue

        print('Parent process %s.' % os.getpid())
        p = Process(target=train, args=(dealer_index,))
        print('Child process will start.')
        p.start()
        p.join()
        print('Child process end.')
