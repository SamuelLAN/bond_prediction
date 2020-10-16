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

import os
import time
import json
import warnings
import numpy as np

warnings.filterwarnings('ignore')

from models.transformer import Model
from lib.utils import output_and_log
from config import path
from config.param import IS_TRAIN
from config.load import group_path, group_name, group_file_name
from config.param import TIME_DIR
from load.load_group_2 import Loader
from preprocess.gen_input_data_new_for_group import gen_inputs
from lib import ml, utils


class Train:
    """ Run the model and estimate its performance """

    MODEL_CLASS = Model
    MODEL_DIR = TIME_DIR

    def __init__(self):
        trace_suffix = 'd_dealers_2015_split_by_date.json'
        group_index = int(group_file_name.split('_')[1])
        get_all = False
        get_individual = False
        filtering_use_cache = True

        input_windows = [5, 10, 15]
        output_windows = [2]
        buy_sell_plan = 0
        with_day_off = False
        use_volume = False

        if get_all:
            group_index = 'all'

        group_file_path = os.path.join(path.ROOT_DIR, 'groups', f'{group_name}.json')

        self.__dictionary = gen_inputs(group_file_path, group_index, trace_suffix, input_windows, output_windows,
                                       with_day_off, buy_sell_plan, use_volume,
                                       _get_all=get_all, _get_individual=get_individual,
                                       use_cache=filtering_use_cache, return_dictionary=True)

        list_idx_bond_id = list(self.__dictionary.items())
        self.__ar_idx = np.array(list(map(lambda x: x[0], list_idx_bond_id)))

        # initialize data instances
        IS_TRAIN = False
        self.__train_load = Loader(group_path, buffer_size=2000, prefix='train')
        self.__test_load = Loader(group_path, prefix='test')

        self.__train_load.start()
        self.__X_train = self.__train_load.generator(Model.params['batch_size'])
        self.__y_train = None
        self.__X_test, self.__y_test = self.__test_load.all()
        self.__input_dim = self.__train_load.input_dim()
        self.__train_size = self.__train_load.size()

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

        train_input = self.__X_train
        test_end_pos = self.__calulate_last_pos(self.__X_test)
        test_input = [self.__X_test, self.__X_test[:, :1], test_end_pos]

        model.train(train_input, self.__y_train, test_input, self.__y_test, self.__train_size)

        encoder_emb_layer = model.model.encoder.embedding
        decoder_emb_layer = model.model.decoder.embedding

        self.__train_load.stop()

        encoder_embs = encoder_emb_layer(self.__ar_idx)
        decoder_embs = decoder_emb_layer(self.__ar_idx)

        print(encoder_embs.shape)
        print(decoder_embs.shape)

        self.display_emb(encoder_embs)

    def display_emb(self, emb):
        emb = np.array(emb)

        emb_path = utils.get_relative_dir('runtime', 'emb', f'emb_for_{group_name}_{group_file_name}.tsv')
        meta_path = utils.get_relative_dir('runtime', 'emb', f'meta_for_{group_name}_{group_file_name}.tsv')

        with open(emb_path, 'wb') as f:
            for i, vec in enumerate(emb):
                f.write(('\t'.join([str(x) for x in vec]) + '\n').encode('utf-8'))

        with open(meta_path, 'wb') as f:
            for word in self.__dictionary.values():
                f.write((str(word) + '\n').encode('utf-8'))


o_train = Train()
o_train.run()

# 2020_04_04_23_04_33
# 2020_04_05_05_41_22
# 2020_04_05_11_06_34
# 2020_04_05_12_57_23

# 2020_04_09_12_33_13
