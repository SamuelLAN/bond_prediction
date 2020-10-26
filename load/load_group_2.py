import os
import time
import copy
import random
import numpy as np
from config import path, load
from lib import utils
import threading


class Loader:
    random_state = 42

    def __init__(self, dir_path, buffer_size=8000, prefix='train', mask_freq=[]):
        assert os.path.exists(dir_path)
        self.__dir_path = dir_path
        self.__prefix = prefix

        self.__mask = None
        if mask_freq:
            _path_mask = utils.get_relative_dir('runtime', 'cache', f'group_{mask_freq[0]}_mask.pkl')
            masks = utils.load_pkl(_path_mask)
            _mask = masks[mask_freq[1]]
            self.__mask = np.expand_dims(_mask, axis=0)

        # get all file paths
        file_list = list(filter(lambda file_name: file_name[:len(prefix)] == prefix, os.listdir(dir_path)))
        self.__file_list = list(map(lambda file_name: os.path.join(dir_path, file_name), file_list))

        random.seed(self.random_state)
        random.shuffle(self.__file_list)

        self.__X = np.array([])
        self.__y = np.array([])
        self.__data = []
        self.__cur_index = 0
        self.__len_files = len(self.__file_list)
        self.__size = None
        self.__buffer_size = buffer_size
        self.__running = True
        self.__has_load_all = False

        dealers = list(map(lambda x: os.path.split(x)[1].split('_')[1], self.__file_list))
        self.dealers = list(set(dealers))

    def start(self):
        thread = threading.Thread(target=self.__load)
        thread.start()
        print('Start thread for loading data ')

        time.sleep(5)
        self.__statistic_generator()

    def stop(self):
        self.__running = False

    def __load(self):
        while self.__running:
            while len(self.__data) < self.__buffer_size:
                file_path = self.__file_list[self.__cur_index]
                self.__cur_index = (self.__cur_index + 1) % self.__len_files

                X_mask, Y = utils.load_pkl(file_path)
                if not isinstance(self.__mask, type(None)):
                    Y = np.array(Y) * self.__mask
                data = list(zip(X_mask, Y))

                random.seed(42)
                random.shuffle(data)

                self.__data += data

            time.sleep(0.1)

        print('Stop thread for loading data ')

    def size(self):
        if self.__size:
            return self.__size

        file_list_for_size = list(map(lambda x: [os.stat(x).st_size, x], self.__file_list))
        max_size = max(list(map(lambda x: x[0], file_list_for_size)))
        len_files = len(file_list_for_size)

        file_list_for_size = list(filter(lambda x: x[0] < max_size, file_list_for_size))
        len_files -= len(file_list_for_size)

        total = 0
        for _, file_path in file_list_for_size:
            tmp_x, tmp_y = utils.load_pkl(file_path)
            total += len(tmp_y)

        self.__size = len_files * 12 + total
        return self.__size

    @staticmethod
    def __calculate_last_pos(x):
        print(f'calculate last pos (before) shape: {x.shape}')
        x = np.sum(x, axis=-1)
        x[x > 0] = 1
        x = np.sum(x, axis=-1)
        print(f'calculate last pos (after) shape: {x.shape}')
        return x

    def generator(self, batch_size):
        while True:
            while len(self.__data) < batch_size:
                time.sleep(0.3)

            data = self.__data[: batch_size]
            X_mask, Y = zip(*data)
            X_mask = np.array(X_mask)
            Y = np.array(Y)
            decode_input = X_mask[:, :1]
            end_pos = self.__calculate_last_pos(X_mask)

            self.__data = self.__data[batch_size:]
            yield (X_mask, decode_input, end_pos), Y

    def all(self, ratio=1.0):
        if self.__has_load_all:
            return self.__X, self.__y

        print(f'Loading all data from {self.__dir_path} ...')

        length = int(len(self.__file_list) * ratio)
        for i, file_path in enumerate(self.__file_list[:length]):
            if i % 2 == 0:
                progress = float(i + 1) / self.__len_files * 100.
                print('\rprogress: %.2f%% ' % progress, end='')

            X_mask, Y = utils.load_pkl(file_path)

            if not isinstance(self.__mask, type(None)):
                Y = np.array(Y) * self.__mask

            self.__X = np.vstack([self.__X, X_mask]) if len(self.__X) else X_mask
            self.__y = np.vstack([self.__y, Y]) if len(self.__y) else Y

        print('\rprogress: 100.0%\nFinish loading ')

        self.__has_load_all = True

        # # add statistics to log
        # self.__statistic()

        np.random.seed(self.random_state)
        data = list(zip(self.__X, self.__y))
        np.random.shuffle(data)

        self.__X, self.__y = zip(*data)
        self.__X = np.array(self.__X)
        self.__y = np.array(self.__y)

        # add statistics to log
        self.__statistic()

        return self.__X, self.__y

    def get_dealer(self, dealer_index):
        file_list = list(filter(lambda x: os.path.split(x)[1].split('_')[1] == dealer_index, self.__file_list))
        X = np.array([])
        Y = np.array([])
        for i, file_path in enumerate(file_list):
            x_mask, y = utils.load_pkl(file_path)

            if not isinstance(self.__mask, type(None)):
                y = np.array(y) * self.__mask

            X = np.vstack([X, x_mask]) if len(X) else np.array(x_mask)
            Y = np.vstack([Y, y]) if len(Y) else np.array(y)
        return np.array(X, dtype=np.int32), np.array(Y, dtype=np.int32)

    def input_dim(self):
        if self.__X.any():
            return self.__X.shape[-1]
        return self.__data[0][0].shape[-1]

    def input_length(self):
        if self.__X.any():
            return self.__X.shape[-2]
        return self.__data[0][0].shape[-2]

    def __statistic(self):
        # calculate the statistics
        tmp_y = copy.deepcopy(self.__y)
        tmp_y[tmp_y <= 0] = 0
        tmp_y[tmp_y > 0] = 1
        label_cardinality = np.sum(tmp_y) / len(tmp_y)
        label_density = np.mean(tmp_y)

        # add the statistic to log
        load.LOG[f'{self.__prefix}_label_cardinality'] = label_cardinality
        load.LOG[f'{self.__prefix}_label_density'] = label_density
        load.LOG[f'{self.__prefix}_size'] = len(self.__y)
        load.LOG['input_length'] = self.input_length()
        load.LOG['voc_size'] = self.input_dim()
        load.LOG['group_dir'] = self.__dir_path

        print('\n---------------------------')
        print(f'{self.__prefix}_label_cardinality: {label_cardinality}')
        print(f'{self.__prefix}_label_cardinality: {label_density}')
        print(f'{self.__prefix}_size: {len(self.__y)}')
        print(f'input_length: {self.input_length()}')
        print(f'voc_size: {self.input_dim()}')
        print('----------------------------\n')

    def __statistic_generator(self):
        # calculate the statistics
        _, Y = zip(*self.__data)
        Y = np.array(Y)
        tmp_y = copy.deepcopy(Y)
        tmp_y[tmp_y <= 0] = 0
        tmp_y[tmp_y > 0] = 1
        label_cardinality = np.sum(tmp_y) / len(tmp_y)
        label_density = np.mean(tmp_y)

        # add the statistic to log
        load.LOG[f'{self.__prefix}_label_cardinality'] = label_cardinality
        load.LOG[f'{self.__prefix}_label_density'] = label_density
        load.LOG[f'{self.__prefix}_size'] = self.size()
        load.LOG['input_length'] = self.input_length()
        load.LOG['voc_size'] = self.input_dim()
        load.LOG['group_dir'] = self.__dir_path

        print('\n---------------------------')
        print(f'{self.__prefix}_label_cardinality: {label_cardinality}')
        print(f'{self.__prefix}_label_cardinality: {label_density}')
        print(f'{self.__prefix}_size: {self.size()}')
        print(f'input_length: {self.input_length()}')
        print(f'voc_size: {self.input_dim()}')
        print('----------------------------\n')


# group_name = 'all'
# group_path = r'D:\Data\share_mine_laptop\community_detection\data\input_data\group_k_means_split_by_date\no_day_off_no_distinguish_buy_sell_use_transaction_count'
# group_path = os.path.join(group_path, group_name)
#
# o_data = Loader(group_path)
#
# o_data.start()
#
# _size = o_data.size()
# bsize = 128
# steps = int(np.ceil(_size * 1. / bsize))
# step = 0
# g = o_data.generator(bsize)
# for tmp in g:
#     tmp_x, tmp_y = tmp
#     # x = tmp_x[0]
#     print('\n-------------------------')
#     for v in tmp_x:
#         print(np.array(v).shape)
#     print(np.array(tmp_y).shape)
#     # print(tmp_y.shape)
#
#     if not np.array(tmp_x[-1]).any():
#         print('end')
#         exit()
#
# #     # if x.shape[0] != tmp_y.shape[0]:
# #     #     print('##################################')
#
#     step += 1
#     if step > steps:
#         break
#
# # X, y = o_data.all()
#
# o_data.stop()
#
# # print(X.shape)
# # print(y.shape)
