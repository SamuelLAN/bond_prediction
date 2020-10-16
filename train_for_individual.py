#!/usr/bin/Python
# -*- coding: utf-8 -*-
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import time
import sys
import json
import warnings
import numpy as np
import os

warnings.filterwarnings('ignore')

from models.transformer_modified_zero import Model
from lib.utils import output_and_log
from config import path
from config.load import LOG, group_path
from config.param import TIME_DIR, measure_dict
from load.load_group_2 import Loader


class Train:
    """ Run the model and estimate its performance """

    MODEL_CLASS = Model
    MODEL_DIR = TIME_DIR

    def __init__(self, _group_path, use_generator=False):
        self.__use_generator = use_generator

        # initialize data instances
        self.__train_load = Loader(_group_path, buffer_size=20000, prefix='train')
        self.__test_load = Loader(_group_path, prefix='test')

        if not self.__use_generator:
            self.__X_train, self.__y_train = self.__train_load.all()
            self.__X_test, self.__y_test = self.__test_load.all()
            self.__input_dim = self.__train_load.input_dim()
            self.__train_size = len(self.__y_train)

        else:
            self.__train_load.start()
            self.__X_train = self.__train_load.generator(Model.params['batch_size'])
            self.__y_train = None
            self.__X_test, self.__y_test = self.__test_load.all()
            self.__input_dim = self.__train_load.input_dim()
            self.__train_size = self.__train_load.size()

        # self.__split_data()

    # def __split_data(self):
    #     """ split data into training set, validation set and test set """
    #     # ready for split data
    #     data_length = len(self.0__X_train_all)
    #     train_end_index = int(TRAIN_VAL_RATIO * data_length)
    #
    #     self.__X_train = self.__X_train_all[: train_end_index]
    #     self.__y_train = self.__y_train_all[: train_end_index]
    #     self.__X_val = self.__X_train_all[train_end_index:]
    #     self.__y_val = self.__y_train_all[train_end_index:]
    #     del self.__X_train_all
    #     del self.__y_train_all

    def __calulate_last_pos(self, x):
        x = np.sum(x, axis=-1)
        x[x > 0] = 1
        x = np.sum(x, axis=-1)
        return x

    def run(self):
        """ train model """
        print('\nStart training model %s/%s ...' % (self.MODEL_DIR, path.TRAIN_MODEL_NAME))

        # initialize model instance
        model = self.MODEL_CLASS(self.__input_dim, self.MODEL_DIR, path.TRAIN_MODEL_NAME, self.__y_test.shape[-1])

        if not self.__use_generator:
            train_end_pos = self.__calulate_last_pos(self.__X_train)
            test_end_pos = self.__calulate_last_pos(self.__X_test)
            train_input = [self.__X_train, self.__X_train[:, :1], train_end_pos]
            test_input = [self.__X_test, self.__X_test[:, :1], test_end_pos]

        else:
            train_input = self.__X_train
            test_end_pos = self.__calulate_last_pos(self.__X_test)
            test_input = [self.__X_test, self.__X_test[:, :1], test_end_pos]

        # train model
        train_start_time = time.time()
        val_result_dict = model.train(train_input, self.__y_train, test_input, self.__y_test, self.__train_size)
        train_use_time = time.time() - train_start_time

        # test model
        eval_train_start_time = time.time()
        train_result_dict = model.test(None, None, train_input, self.__y_train, None, 'train', self.__train_size)
        eval_train_end_time = time.time()

        test_result_dict = model.test(train_input, self.__y_train, test_input, self.__y_test, None, 'test')
        eval_test_use_time = time.time() - eval_train_end_time
        eval_train_time = eval_train_end_time - eval_train_start_time

        print('\nFinish training\n')

        dealers_result_dict = {}
        # test_dealers = self.__test_load.dealers
        # len_dealers = len(test_dealers)
        # for i, dealer_index in enumerate(test_dealers):
        #     if i % 2 == 0:
        #         progress = float(i + 1) / len_dealers * 100.
        #         print('\rprogress: %.2f%% ' % progress, end='')
        #
        #     batch_x, batch_y = self.__test_load.get_dealer(dealer_index)
        #     batch_end_pos = self.__calulate_last_pos(batch_x)
        #     batch_input = [batch_x, batch_x[:, :1], batch_end_pos]
        #
        #     tmp_dealer_result_dict = model.test_in_batch(batch_input, batch_y,
        #                                                  name=f'dealer_{dealer_index}', data_size=len(batch_x))
        #     dealers_result_dict[dealer_index] = tmp_dealer_result_dict

        # show results
        self.__log_results(self.MODEL_DIR, train_result_dict, val_result_dict, test_result_dict,
                           self.MODEL_CLASS.params, train_use_time, eval_train_time, eval_test_use_time,
                           dealers_result_dict)

        if self.__use_generator:
            self.__train_load.stop()

    @staticmethod
    def __log_results(model_time, train_result_dict, val_result_dict, test_result_dict,
                      model_params, train_use_time, eval_train_time, eval_test_use_time, dealer_result_dict):
        """
        Show the validation result
         as well as the model params to console and save them to the log file.
        """

        dealer_results = 'self'
        # dealer_results = list(map(lambda x: f'{x[0]}: {x[1]}', dealer_result_dict.items()))
        # dealer_results = '\n'.join(dealer_results)

        data = (path.MODEL_NAME,
                model_time,
                train_result_dict,
                test_result_dict,
                model_params,
                str(train_use_time),
                str(eval_train_time),
                str(eval_test_use_time),
                json.dumps(LOG),
                time.strftime('%Y.%m.%d %H:%M:%S'),
                dealer_results)

        output = 'model_name: %s\n' \
                 'model_time: %s\n' \
                 'train_result_dict: %s\n' \
                 'test_result_dict: %s\n' \
                 'model_params: %s\n' \
                 'train_use_time: %ss\n' \
                 'eval_train_time: %ss\n' \
                 'eval_test_use_time: %ss\n' \
                 'data_param: %s\n' \
                 'time: %s\n' \
                 'dealer_results:\n%s\n\n' % data

        # show and then save result to log file
        output_and_log(path.PATH_MODEL_LOG, output)

        model_name = path.MODEL_NAME
        data_statistics = ','.join(list(map(
            lambda x: str(round(x, 4)) if isinstance(x, float) else str(x),
            [
                LOG['group_name'],
                LOG['group_param_name'],
                LOG['group_file_name'],
                LOG['train_label_cardinality'],
                LOG['test_label_cardinality'],
                LOG['train_label_density'],
                LOG['test_label_density'],
                LOG['voc_size'],
                LOG['input_length'],
                LOG['train_size'],
                LOG['test_size'],
            ])))

        csv_headline = 'model_name,group_name,group_param_name,group_file_name'
        csv_headline += ',train_label_cardinality,test_label_cardinality,train_label_density,test_label_density'
        csv_headline += ',voc_size,input_length,train_size,test_size'
        csv_headline += ',' + ','.join(list(map(lambda x: f'train_{x[0]}', measure_dict.items())))
        csv_headline += ',' + ','.join(list(map(lambda x: f'test_{x[0]}', measure_dict.items()))) + '\n'

        csv_string = f'{model_name},{data_statistics}'
        for k, v in train_result_dict.items():
            csv_string += f',{round(v, 4)}'
        for k, v in test_result_dict.items():
            csv_string += f',{round(v, 4)}'

        output_and_log(path.PATH_CSV_LOG, csv_string, csv_headline)

# if __name__ == '__main__':
#
#     group_dir = os.path.split(group_path)[0]
#     dir_list = list(filter(lambda x: x[0].lower() == 'i', os.listdir(group_dir)))
#     len_dir = len(dir_list)
#
#     for i, dealer_index in enumerate(dir_list):
#         if dealer_index < 'i1156':
#             continue
#
#         print('\n\n--------------------------------------------------')
#         print(f'training model for ### {dealer_index} ### ')
#         print('---------------------------------------------------\n')
#
#         LOG['group_file_name'] = dealer_index
#         Train.MODEL_DIR = TIME_DIR = time.strftime('%Y_%m_%d_%H_%M_%S')
#
#         tmp_group_path = os.path.join(group_dir, dealer_index)
#         o_train = Train(_group_path=tmp_group_path, use_generator=False)
#         o_train.run()
#         del o_train
#
#     # 2020_04_04_23_04_33
#     # 2020_04_05_05_41_22
