import os
import time
import copy
import numpy as np
from config import path, load
from lib import utils
import threading


class Loader:
    def __init__(self, dir_path, buffer_size=8000):
        assert os.path.exists(dir_path)
        self.__dir_path = dir_path
        self.__is_train_set = True if 'train' in dir_path.lower() else False
        self.__file_list = list(map(lambda file_name: os.path.join(dir_path, file_name), os.listdir(dir_path)))
        self.__X = np.array([])
        self.__y = np.array([])
        self.__cur_index = 0
        self.__len_files = len(self.__file_list)
        self.__buffer_size = buffer_size
        self.__running = True
        self.__has_load_all = False

    def start(self):
        thread = threading.Thread(target=self.__load)
        thread.start()
        print('Start thread for loading data ')

    def stop(self):
        self.__running = False

    def __load(self):
        while self.__running:

            while len(self.__X) < self.__buffer_size:
                file_path = self.__file_list[self.__cur_index]
                self.__cur_index = (self.__cur_index + 1) % self.__len_files

                X_mask, Y = utils.load_pkl(file_path)

                self.__X = np.vstack([self.__X, X_mask]) if len(self.__X) else X_mask
                self.__y = np.vstack([self.__y, Y]) if len(self.__y) else Y

            time.sleep(1)

        print('Stop thread for loading data ')

    def next_batch(self):
        pass

    def all(self):
        if self.__has_load_all:
            return self.__X, self.__y

        print(f'Loading all data from {self.__dir_path} ...')

        for i, file_path in enumerate(self.__file_list):
            if i % 2 == 0:
                progress = float(i + 1) / self.__len_files * 100.
                print('\rprogress: %.2f%% ' % progress, end='')

            X_mask, Y = utils.load_pkl(file_path)
            self.__X = np.vstack([self.__X, X_mask]) if len(self.__X) else X_mask
            self.__y = np.vstack([self.__y, Y]) if len(self.__y) else Y

        print('\rprogress: 100.0%%\nFinish loading ')

        self.__has_load_all = True

        # add statistics to log
        self.__statistic()

        return self.__X, self.__y

    def input_dim(self):
        return self.__X.shape[-1]

    def input_length(self):
        return self.__X.shape[-2]

    def __statistic(self):
        # calculate the statistics
        tmp_y = copy.deepcopy(self.__y)
        tmp_y[tmp_y <= 0] = 0
        tmp_y[tmp_y > 0] = 1
        label_cardinality = np.sum(tmp_y) / len(tmp_y)
        label_density = np.mean(tmp_y)

        # add the statistic to log
        prefix = 'train' if self.__is_train_set else 'test'
        load.LOG[f'{prefix}_label_cardinality'] = label_cardinality
        load.LOG[f'{prefix}_label_density'] = label_density
        load.LOG[f'{prefix}_size'] = len(self.__y)
        load.LOG['input_length'] = self.input_length()
        load.LOG['voc_size'] = self.input_dim()
        load.LOG['group_dir'] = self.__dir_path

        print('\n---------------------------')
        print(f'{prefix}_label_cardinality: {label_cardinality}')
        print(f'{prefix}_label_cardinality: {label_density}')
        print(f'{prefix}_size: {len(self.__y)}')
        print(f'input_length: {self.input_length()}')
        print(f'voc_size: {self.input_dim()}')
        print('----------------------------\n')


# group_name = 'group_2_no_below_50_25_10_g_minus_1_1_train'
# group_path = r'D:\Data\share_mine_laptop\community_detection\data\inputs\group_Spectral_Clustering_filter_lower_5_with_model_input_features\no_day_off_no_distinguish_buy_sell_use_transaction_count'
# group_path = os.path.join(group_path, group_name)
#
# o_data = Loader(group_path)
# X, y = o_data.all()
#
# print(X.shape)
# print(y.shape)
