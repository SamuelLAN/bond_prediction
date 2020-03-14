#!/usr/bin/Python
# -*- coding: utf-8 -*-
import time
import json
import warnings

warnings.filterwarnings('ignore')

from models.lstm_new import Model
from lib.utils import output_and_log
from config import path
from config.load import LOG
from config.param import TIME_DIR
from load.load_group import Loader


class Train:
    """ Run the model and estimate its performance """

    TRAIN_GROUP = r'D:\Data\share_mine_laptop\community_detection\data\inputs\group_Spectral_Clustering_filter_lower_5_with_model_input_features\no_day_off_no_distinguish_buy_sell_use_transaction_count\group_2_no_below_50_25_10_g_minus_1_1_train'
    TEST_GROUP = r'D:\Data\share_mine_laptop\community_detection\data\inputs\group_Spectral_Clustering_filter_lower_5_with_model_input_features\no_day_off_no_distinguish_buy_sell_use_transaction_count\group_2_no_below_50_25_10_g_minus_1_1_test'

    MODEL_CLASS = Model
    MODEL_DIR = TIME_DIR

    def __init__(self):
        # initialize data instances
        train_load = Loader(self.TRAIN_GROUP)
        test_load = Loader(self.TEST_GROUP)
        self.__X_train, self.__y_train = train_load.all()
        self.__X_test, self.__y_test = test_load.all()
        self.__input_dim = train_load.input_dim()

        # self.__topic_mask = o_load.topic_mask()
        # self.__split_data()

    # def __split_data(self):
    #     """ split data into training set, validation set and test set """
    #     # ready for split data
    #     data_length = len(self.__X_train_all)
    #     train_end_index = int(TRAIN_VAL_RATIO * data_length)
    #
    #     self.__X_train = self.__X_train_all[: train_end_index]
    #     self.__y_train = self.__y_train_all[: train_end_index]
    #     self.__X_val = self.__X_train_all[train_end_index:]
    #     self.__y_val = self.__y_train_all[train_end_index:]
    #     del self.__X_train_all
    #     del self.__y_train_all

    def run(self):
        """ train model """
        print('\nStart training model %s/%s ...' % (self.MODEL_DIR, path.TRAIN_MODEL_NAME))

        # initialize model instance
        model = self.MODEL_CLASS(self.__input_dim, self.MODEL_DIR, path.TRAIN_MODEL_NAME, self.__y_train.shape[-1])

        # train model
        train_start_time = time.time()
        val_result_dict = model.train(self.__X_train, self.__y_train, self.__X_test, self.__y_test)
        train_use_time = time.time() - train_start_time

        # test model
        eval_train_start_time = time.time()
        train_result_dict = model.test(None, None, self.__X_train, self.__y_train, None, 'train')
        # train_result_dict_with_mask = model.test(None, None, self.__X_train, self.__y_train, None, 'train')
        eval_train_end_time = time.time()
        test_result_dict = model.test(self.__X_train, self.__y_train, self.__X_test, self.__y_test, None, 'test')
        # test_result_dict_with_mask = model.test(self.__X_train, self.__y_train, self.__X_test, self.__y_test,
        #                                         name='test')
        eval_test_use_time = time.time() - eval_train_end_time
        eval_train_time = eval_train_end_time - eval_train_start_time

        print('\nFinish training\n')

        # show results
        self.__log_results(self.MODEL_DIR, train_result_dict, val_result_dict, test_result_dict,
                           self.MODEL_CLASS.params, train_use_time, eval_train_time, eval_test_use_time)

    @staticmethod
    def __log_results(model_time, train_result_dict, val_result_dict, test_result_dict,
                      model_params, train_use_time, eval_train_time, eval_test_use_time):
        """
        Show the validation result
         as well as the model params to console and save them to the log file.
        """

        data = (path.MODEL_NAME,
                model_time,
                train_result_dict,
                test_result_dict,
                model_params,
                str(train_use_time),
                str(eval_train_time),
                str(eval_test_use_time),
                json.dumps(LOG),
                time.strftime('%Y.%m.%d %H:%M:%S'))

        output = 'model_name: %s\n' \
                 'model_time: %s\n' \
                 'train_result_dict: %s\n' \
                 'test_result_dict: %s\n' \
                 'model_params: %s\n' \
                 'train_use_time: %ss\n' \
                 'eval_train_time: %ss\n' \
                 'eval_test_use_time: %ss\n' \
                 'data_param: %s\n' \
                 'time: %s\n\n' % data

        # show and then save result to log file
        output_and_log(path.PATH_MODEL_LOG, output)

        import os
        from config import load as load_conf

        model_name = path.MODEL_NAME
        subset = LOG['subset']
        volume_level = LOG['volume_level']
        data_index = LOG['data_index']
        data_statistics = ','.join(list(map(lambda x: str(round(x, 2)), [
            LOG['train_label_cardinality'],
            LOG['test_label_cardinality'],
            LOG['train_label_density'],
            LOG['test_label_density'],
            LOG['voc_size'],
        ])))

        csv_string = f'{model_name},{subset},{volume_level},{data_index},{data_statistics}'
        for k, v in train_result_dict.items():
            csv_string += f',{round(v, 4)}'
        for k, v in test_result_dict.items():
            csv_string += f',{round(v, 4)}'

        output_and_log(path.PATH_CSV_LOG, csv_string)


o_train = Train()
o_train.run()
