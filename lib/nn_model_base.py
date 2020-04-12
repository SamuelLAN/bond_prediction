#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import re
import math
import types
import numpy as np
import tensorflow as tf
from config.param import IS_TRAIN, measure_dict, RANDOM_STATE, NEW_TIME_DIR
from config.path import PATH_MODEL_DIR, PATH_BOARD_DIR, mkdir_time
from tf_callback.saver import Saver
from tf_callback.board import Board

keras = tf.compat.v1.keras


class NN:
    default_params = {
        'learning_rate': 1e-7,
        'lr_decay_rate': 0.001,
        'lr_staircase': False,
        'lr_factor': 0.1,
        'lr_patience': 3,
        'batch_size': 5,
        'epoch': 100,
        'early_stop': 30,
        'auto_open_tensorboard': True,
        'histogram_freq': 0,
        'update_freq': 'epoch',
        'monitor': 'val_categorical_accuracy',
        'monitor_mode': 'max',
        'monitor_start_train': 'categorical_accuracy',
        'monitor_start_train_val': 0.75,
        'initial_epoch': 0,
        'random_state': RANDOM_STATE,
    }

    # model param config
    params = {**default_params}

    @property
    def config_for_keras(self):
        """ NEED: Customize the config for keras """
        return {
            'optimizer': tf.compat.v1.train.AdamOptimizer,
            # 'loss': keras.losses.binary_crossentropy,
            'loss': 'sparse_categorical_crossentropy',
            'metrics': [
                keras.metrics.categorical_accuracy,
                keras.metrics.categorical_crossentropy,
            ],
            'callbacks': [
                self.callback_tf_board,
                self.callback_saver,
                # self.callback_reduce_lr,
            ],
        }

    def __init__(self, model_dir, model_name=None, num_classes=None):
        self.time_dir = model_dir
        self.num_classes = num_classes
        self.__model_dir = mkdir_time(PATH_MODEL_DIR, model_dir)
        self.__update_model_dir = mkdir_time(PATH_MODEL_DIR, NEW_TIME_DIR)
        self.__monitor_bigger_best = self.params['monitor_mode'] == 'max'

        # initialize some variables that would be used by func "model.fit";
        #   the child class can change this params when customizing the build func
        self.__class_weight = None
        self.__initial_epoch = 0 if 'initial_epoch' not in self.params else self.params['initial_epoch']

        # get the tensorboard dir path
        self.__get_tf_board_path(model_dir)

        # get the model path
        self.__get_model_path(model_name)

        self.init_gpu_config()

        # build model
        self.build()

        # initialize some callback funcs
        self.__init_callback()

    def build(self):
        """ Build neural network architecture; Need to customize """
        self.model = None

    @staticmethod
    def init_gpu_config():
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

    def __get_model_path(self, model_name):
        """ Get the model path """
        self.model_path = os.path.join(self.__model_dir, model_name + '.hdf5')
        self.checkpoint_path = os.path.join(self.__model_dir,
                                            model_name + '.{epoch:03d}-{%s:.4f}.hdf5' % self.params['monitor'])

        self.__update_model_path = os.path.join(self.__update_model_dir, model_name + '.hdf5')
        self.__update_checkpoint_path = os.path.join(self.__update_model_dir,
                                                     model_name + '.{epoch:03d}-{%s:.4f}.hdf5' % self.params['monitor'])

        # check if model exists
        if not os.path.isfile(self.model_path) and not os.path.isfile(self.model_path + '.index'):
            model_path = self.__get_best_model_path()
            if model_path:
                self.model_path = model_path

    def test_func(self):
        self.__get_best_model_path()

    def __get_best_model_path(self):
        """ Return the best model in model_dir """
        # check if any model exists
        if not os.listdir(self.__model_dir):
            return

        # initialize some variables
        best = -np.inf if self.__monitor_bigger_best else np.inf
        best_epoch = 0
        best_file_name = ''
        reg = re.compile('\.(\d+)-(\d+\.\d+)\.hdf5')

        # check all the model name in model dir
        for file_name in os.listdir(self.__model_dir):
            # filter irrelevant files
            if '.hdf5' != file_name[-len('.hdf5'):] and '.hdf5.index' != file_name[-len('.hdf5.index'):]:
                continue

            epoch, monitor = reg.findall(file_name)[0]
            epoch = int(epoch)
            monitor = float(monitor)

            # compare the result, find out if it is the best
            if (self.__monitor_bigger_best and best < monitor) or (not self.__monitor_bigger_best and best > monitor) \
                    or (best == monitor and best_epoch < epoch):
                best = monitor
                best_file_name = file_name.replace('.hdf5', '').replace('.index', '')
                best_epoch = epoch

        return os.path.join(self.__model_dir, best_file_name + '.hdf5')

    def __get_tf_board_path(self, model_dir):
        """ Get the tensorboard dir path and run it on cmd """
        self.tf_board_dir = mkdir_time(PATH_BOARD_DIR, model_dir)
        self.__update_tf_board_dir = mkdir_time(PATH_BOARD_DIR, NEW_TIME_DIR)

    def set_learning_rate(self, lr):
        self.__learning_rate = lr

    def __init_variables(self, data_size):
        """ Initialize some variables that will be used while training """
        self.__global_step = tf.compat.v1.train.get_or_create_global_step()
        self.__steps_per_epoch = int(np.ceil(data_size * 1. / self.params['batch_size']))
        self.__steps = self.__steps_per_epoch * self.params['epoch']

        self.__decay_steps = self.__steps if not self.params['lr_staircase'] else self.__steps_per_epoch

        # def decayed_learning_rate(step):
        #     step = min(step, self.__decay_steps)
        #     return ((self.params['learning_rate'] - 1e-10) *
        #             (1 - step / self.__decay_steps) ^ (2)
        #             ) + 1e-10

        self.__learning_rate = self.params['learning_rate']
        # self.__learning_rate = tf.train.exponential_decay(self.params['learning_rate'], self.__global_step,
        #                                                   self.__decay_steps, self.params['lr_decay_rate'],
        #                                                   self.params['lr_staircase'])
        # self.__learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(self.params['learning_rate'],
        #                                                                       decay_steps=self.__decay_steps,
        #                                                                       decay_rate=self.params['lr_decay_rate'],
        #                                                                       staircase=self.params['lr_staircase'])

    def __init_callback(self):
        """ Customize some callbacks """
        self.callback_tf_board = Board(log_dir=self.tf_board_dir,
                                       histogram_freq=self.params['histogram_freq'],
                                       write_grads=False,
                                       write_graph=True,
                                       write_images=False,
                                       profile_batch=0,
                                       update_freq=self.params['update_freq'])
        self.callback_tf_board.set_model(self.model)

        self.callback_saver = Saver(self.checkpoint_path,
                                    self.params['monitor'],
                                    self.params['monitor_mode'],
                                    self.params['early_stop'],
                                    self.params['monitor_start_train'],
                                    self.params['monitor_start_train_val'])
        self.callback_saver.set_model(self.model)

        self.callback_reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                                    factor=self.params['lr_factor'],
                                                                    patience=self.params['lr_patience'],
                                                                    verbose=1)

    def before_train(self, data_size, x, y):
        self.__init_variables(data_size)

        self.compile(self.__learning_rate)

        # initialize global variables
        # keras.backend.get_session().run(tf.compat.v1.global_variables_initializer())

        # if model exists, load the model weight
        if os.path.isfile(self.model_path) or os.path.isfile(self.model_path + '.index'):
            self.load_model(self.model_path, x, y)

            # update paths and callbacks
            self.model_path = self.__update_model_path
            self.checkpoint_path = self.__update_checkpoint_path
            self.tf_board_dir = self.__update_tf_board_dir
            self.__init_callback()

    def train(self, train_x, train_y_one_hot, val_x, val_y_one_hot, train_size, mask=None):
        """ Train model with all data loaded in memory """
        self.before_train(train_size, train_x, train_y_one_hot)

        if IS_TRAIN:
            # The returned value may be useful in the future
            batch_size = self.params['batch_size'] if not isinstance(train_x, types.GeneratorType) else None
            steps_per_epoch = None if not isinstance(train_x, types.GeneratorType) else self.__steps_per_epoch
            history_object = self.model.fit(train_x, train_y_one_hot,
                                            epochs=self.params['epoch'],
                                            batch_size=batch_size,
                                            steps_per_epoch=steps_per_epoch,
                                            validation_data=(val_x, val_y_one_hot),
                                            callbacks=self.config_for_keras['callbacks'],
                                            class_weight=self.__class_weight,
                                            initial_epoch=self.__initial_epoch,
                                            verbose=2)

            # load the best model, then it could be tested
            self.load_model()

        return self.test(train_x, train_y_one_hot, val_x, val_y_one_hot, mask)

    @staticmethod
    def reset_graph():
        keras.backend.get_session().close()
        tf.compat.v1.reset_default_graph()
        keras.backend.set_session(tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph()))

    def compile(self, learning_rate):
        self.model.compile(optimizer=self.config_for_keras['optimizer'](learning_rate=learning_rate),
                           loss=self.config_for_keras['loss'],
                           metrics=self.config_for_keras['metrics'])

    def load_model(self, model_path='', x=None, y=None):
        # auto find best model path
        if not model_path:
            model_path = self.__get_best_model_path()
            if not model_path:
                return

        # empty fit, to prevent error from occurring when loading model
        if not self.model.built and not isinstance(x, type(None)):
            if isinstance(x, types.GeneratorType):
                for tmp_x, tmp_y in x:
                    if isinstance(tmp_x, tuple):
                        x = tuple([v[:1] for v in tmp_x])
                    else:
                        x = tmp_x[:1]
                    y = tmp_y[:1]
                    break
            self.model.fit(x, y, epochs=0, verbose=0)

        self.model.load_weights(model_path)
        print('Finish loading weights from %s ' % model_path)

    def test(self, train_x, train_y_one_hot, val_x, val_y_one_hot, mask=None, name='val', data_size=None):
        """ Customize for testing model """
        if name == 'train':
            return self.test_in_batch(val_x, val_y_one_hot, mask, name, data_size)

        # evaluate the validation data
        return self.measure_and_print(val_y_one_hot, self.predict(val_x), mask, name, {})

    def test_in_batch(self, x, y_ont_hot, mask=None, name='val', data_size=None):
        """ evaluate the model performance while data size is big """
        # variables that record all results
        logits_list = []

        # calculate the total steps
        batch_size = self.params['batch_size']
        data_size = data_size if data_size else len(y_ont_hot)
        steps = int(math.ceil(data_size * 1.0 / batch_size))

        # traverse all data
        # TODO change here
        if isinstance(x, types.GeneratorType):
            step = 0
            result_dict = {}

            for tmp in x:
                tmp_inputs, tmp_y = tmp
                logits = self.predict(tmp_inputs)
                result_dict = self.measure_and_print(tmp_y, logits, name=name, result_dict=result_dict, show=False, sum=True)

                if step % 5 == 0:
                    progress = float(step + 1) / steps * 100.
                    print('\rprogress: %.2f%% ' % progress, end='')

                step += 1
                if step >= steps:
                    break

            print('\n-----------------------------------------')
            for key, val in result_dict.items():
                result_dict[key] = val / float(steps)
                print('%s %s: %f' % (name, key, result_dict[key]))

            return result_dict

        else:
            for step in range(steps):
                if isinstance(x, list):
                    tmp_x = [v[step * batch_size: (step + 1) * batch_size] for v in x]
                else:
                    tmp_x = x[step * batch_size: (step + 1) * batch_size]
                logits_list.append(self.predict(tmp_x))

        logits_list = np.vstack(logits_list)

        return self.measure_and_print(y_ont_hot, logits_list, mask, name, {})

    @staticmethod
    def measure_and_print(y_ont_hot, logits_list, mask=None, name='val', result_dict={}, show=True, sum=False):
        logits_list = logits_list.reshape(y_ont_hot.shape)  # make the shape of logits the same as y_true

        for key, func in measure_dict.items():
            val = func(y_ont_hot, logits_list, mask)
            if key in result_dict and sum:
                result_dict[key] += val
            else:
                result_dict[key] = val

        # show results
        if show:
            print('\n-----------------------------------------')
            for key, value in result_dict.items():
                print('%s %s: %f' % (name, key, value))

        return result_dict

    def predict(self, x):
        return self.model.predict(x)

    def predict_class(self, x):
        output = self.predict(x)
        return np.argmax(output, axis=-1)

    def predict_correct(self, x, y_one_hot):
        prediction = self.predict_class(x)
        y = np.argmax(y_one_hot, axis=-1)
        return prediction == y

    def predict_prob(self, x, class_index=-1):
        return self.predict(x)[:, class_index]

    def save(self):
        self.model.save_weights(self.model_path, save_format='h5')
        print("Finish saving model to %s " % self.model_path)

    def gen_model_from_layer(self, start_layer_index, end_layer_index, with_input=False, input_shape=None):
        model_list = self.model.layers[start_layer_index: end_layer_index + 1]
        if with_input:
            model_list = [keras.layers.InputLayer(input_shape, dtype=tf.float32)] + model_list
        return keras.Sequential(model_list)

    def get_weight_from_name(self, name):
        weights = self.model.weights
        for weight in weights:
            if weight.name != name:
                continue
            return weight
