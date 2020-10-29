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
import json
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# from models.transformer import Model
# from models.fc import Model
# from models.lstm import Model
# from models.bilstm import Model
# from models.transformer_standard import Model
# from models.transformer_modified_dense import Model
# from models.transformer_modified_dense_2 import Model
from models.transformer_modified_zero import Model
# from models.transformer_rezero import Model
from lib.utils import output_and_log
from config import path
from config.load import LOG, group_path_dealer_prediction, group_file_name, freq_level
from config.param import TIME_DIR, measure_dict
from load.load_group_2 import Loader


class Train:
    """ Run the model and estimate its performance """

    MODEL_CLASS = Model
    MODEL_DIR = TIME_DIR

    def __init__(self, use_generator=False):
        self.__use_generator = use_generator

        # from lib import utils
        # group_index = group_file_name.split('_')[1]
        # _path_mask = utils.get_relative_dir('runtime', 'cache', f'group_{group_index}_mask.pkl')
        # masks = utils.load_pkl(_path_mask)
        # _mask = masks[freq_level]
        # self.__mask = np.expand_dims(_mask, axis=0)
        self.__mask = None

        # initialize data instances
        # print(group_path)
        self.__train_load = Loader(group_path_dealer_prediction, buffer_size=2000, prefix='train',
                                   # mask_freq=[group_index, freq_level]
                                   )
        self.__test_load = Loader(group_path_dealer_prediction, prefix='test',
                                  # mask_freq=[group_index, freq_level]
                                  )

        if not self.__use_generator:
            self.__X_train, self.__y_train = self.__train_load.all()
            self.__X_test, self.__y_test = self.__test_load.all()
            self.__input_dim = self.__train_load.input_dim()
            self.__train_size = len(self.__y_train)

        else:
            self.__train_load.start()
            self.__X_train = self.__train_load.generator(Model.params['batch_size'])
            self.__y_train = None
            self.__X_test, self.__y_test = self.__test_load.all(0.1)
            self.__input_dim = self.__train_load.input_dim()
            self.__train_size = self.__train_load.size()

        # voc_size = int((self.__input_dim - 2) / 2)
        # self.__y_test_buy_mask = np.array([[1] * voc_size + [0] * voc_size + [0] * 2], dtype=np.float32)
        # self.__y_test_sell_mask = np.array([[0] * voc_size + [1] * voc_size + [0] * 2], dtype=np.float32)
        # self.__y_test_buy_mask = self.__y_test_sell_mask
        self.__y_test_buy_mask = self.__y_test_sell_mask = None

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

        # train_input = self.__X_train
        # test_input = self.__X_test

        # train model
        train_start_time = time.time()
        val_result_dict = model.train(train_input, self.__y_train, test_input, self.__y_test, self.__train_size)
        train_use_time = time.time() - train_start_time

        # model.analyze()

        # test model
        eval_train_start_time = time.time()
        train_result_dict = model.test(None, None, train_input, self.__y_train, self.__mask, 'train', self.__train_size)
        eval_train_end_time = time.time()

        test_result_dict = model.test(train_input, self.__y_train, test_input, self.__y_test, self.__y_test_buy_mask, 'test')
        eval_test_use_time = time.time() - eval_train_end_time
        eval_train_time = eval_train_end_time - eval_train_start_time

        print('\nFinish training\n')

        dealers_result_dict = {}
        test_dealers = self.__test_load.dealers
        len_dealers = len(test_dealers)
        for i, dealer_index in enumerate(test_dealers):
            if i % 2 == 0:
                progress = float(i + 1) / len_dealers * 100.
                print('\rprogress: %.2f%% ' % progress, end='')

            batch_x, batch_y = self.__test_load.get_dealer(dealer_index)
            batch_end_pos = self.__calulate_last_pos(batch_x)
            batch_input = [batch_x, batch_x[:, :1], batch_end_pos]
            # batch_input = batch_x

            tmp_dealer_result_dict = model.test_in_batch(batch_input, batch_y, mask=self.__y_test_buy_mask,
                                                         name=f'dealer_{dealer_index}', data_size=len(batch_x))
            dealers_result_dict[dealer_index] = tmp_dealer_result_dict

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

        dealer_results = list(map(lambda x: f'{x[0]}: {x[1]}', dealer_result_dict.items()))
        dealer_results = '\n'.join(dealer_results)

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
                 'bond_results:\n%s\n\n' % data

        # show and then save result to log file
        output_and_log(path.PATH_MODEL_LOG_DEALER, output)

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

        output_and_log(path.PATH_CSV_LOG_DEALER, csv_string, csv_headline)


o_train = Train(use_generator=True)
o_train.run()

# 2020_04_04_23_04_33
# 2020_04_05_05_41_22
# 2020_04_05_11_06_34
# 2020_04_05_12_57_23

# 2020_04_09_12_33_13


# 5 days  group 0
# Save model to D:\Github\bond_prediction_branches\dev_2.0\bond_prediction\runtime\models\transformer_modified_zero_with_summed_bond_embeddings_for_5_days_input\2020_06_28_02_44_25\transformer_modified_zero_with_summed_bond_embeddings_for_5_days_input.226-0.6943.hdf5
# 13443/13443 - 10s - loss: 0.0014 - tf_accuracy: 0.9996 - tf_hamming_loss: 3.6600e-04 - tf_f1: 0.9616 - tf_precision: 0.9837 - tf_recall: 0.9406 - val_loss: 0.0075 - val_tf_accuracy: 0.9978 - val_tf_hamming_loss: 0.0022 - val_tf_f1: 0.6943 - val_tf_precision: 0.7798 - val_tf_recall: 0.6260
# Epoch 227/3000

# 5 days group 1
# Epoch 260/3000
# Save model to D:\Github\bond_prediction_branches\dev_2.0\bond_prediction\runtime\models\transformer_modified_zero_with_summed_bond_embeddings_for_5_days_input\2020_06_28_11_31_34\transformer_modified_zero_with_summed_bond_embeddings_for_5_days_input.260-0.5770.hdf5
# 12606/12606 - 11s - loss: 0.0139 - tf_accuracy: 0.9956 - tf_hamming_loss: 0.0044 - tf_f1: 0.8598 - tf_precision: 0.8894 - tf_recall: 0.8323 - val_loss: 0.0340 - val_tf_accuracy: 0.9893 - val_tf_hamming_loss: 0.0107 - val_tf_f1: 0.5770 - val_tf_precision: 0.6342 - val_tf_recall: 0.5295
# Epoch 261/3000

# 5 days group 2
# Epoch 116/3000
# Save model to D:\Github\bond_prediction_branches\dev_2.0\bond_prediction\runtime\models\transformer_modified_zero_with_summed_bond_embeddings_for_5_days_input\2020_06_28_12_31_54\transformer_modified_zero_with_summed_bond_embeddings_for_5_days_input.116-0.6133.hdf5
# 5668/5668 - 5s - loss: 0.1371 - tf_accuracy: 0.9502 - tf_hamming_loss: 0.0498 - tf_f1: 0.7248 - tf_precision: 0.7440 - tf_recall: 0.7068 - val_loss: 0.1703 - val_tf_accuracy: 0.9352 - val_tf_hamming_loss: 0.0648 - val_tf_f1: 0.6133 - val_tf_precision: 0.6225 - val_tf_recall: 0.6046
# Epoch 117/3000

# 5 days group 3
# Epoch 257/3000
# Save model to D:\Github\bond_prediction_branches\dev_2.0\bond_prediction\runtime\models\transformer_modified_zero_with_summed_bond_embeddings_for_5_days_input\2020_06_28_12_55_32\transformer_modified_zero_with_summed_bond_embeddings_for_5_days_input.257-0.8153.hdf5
# 11111/11111 - 7s - loss: 0.0016 - tf_accuracy: 0.9996 - tf_hamming_loss: 4.1629e-04 - tf_f1: 0.9484 - tf_precision: 0.9950 - tf_recall: 0.9062 - val_loss: 0.0044 - val_tf_accuracy: 0.9988 - val_tf_hamming_loss: 0.0012 - val_tf_f1: 0.8153 - val_tf_precision: 0.9244 - val_tf_recall: 0.7298
# Epoch 258/3000

# 5 days group 0
# Epoch 266/3000
# Save model to D:\Github\bond_prediction_branches\dev_2.0\bond_prediction\runtime\models\transformer_modified_zero_with_summed_bond_embeddings_for_10_days_input\2020_06_28_13_29_50\transformer_modified_zero_with_summed_bond_embeddings_for_10_days_input.266-0.7029.hdf5
# 13443/13443 - 10s - loss: 0.0011 - tf_accuracy: 0.9997 - tf_hamming_loss: 3.0531e-04 - tf_f1: 0.9682 - tf_precision: 0.9858 - tf_recall: 0.9514 - val_loss: 0.0072 - val_tf_accuracy: 0.9979 - val_tf_hamming_loss: 0.0021 - val_tf_f1: 0.7029 - val_tf_precision: 0.8145 - val_tf_recall: 0.6190
# Epoch 267/3000



