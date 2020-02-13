#!/usr/bin/Python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

keras = tf.keras


class Saver(keras.callbacks.Callback):
    def __init__(self, file_path, monitor, mode, early_stop,
                 start_train_monitor='categorical_accuracy', start_train_monitor_val=0.65):
        super(Saver, self).__init__()
        self.__file_path = file_path
        self.__monitor = monitor
        self.__mode = mode
        self.__early_stop = early_stop
        self.__start_train_monitor = start_train_monitor
        self.__start_train_monitor_val = start_train_monitor_val

        self.__patience = 0
        self.__best = -np.Inf if self.__mode == 'max' else np.Inf

    def on_epoch_end(self, epoch, logs=None):
        if self.__start_train_monitor in logs and logs[self.__start_train_monitor] < self.__start_train_monitor_val:
            return

        monitor = logs[self.__monitor]

        if (self.__mode == 'max' and monitor >= self.__best) or (self.__mode == 'min' and monitor <= self.__best):
            filepath = self.__file_path.format(epoch=epoch + 1, **logs)
            self.model.save_weights(filepath, overwrite=True)
            self.__best = monitor
            self.__patience = 0
            print('Save model to %s' % filepath)

        else:
            self.__patience += 1
            if self.__patience > self.__early_stop:
                self.model.stop_training = True
                print("Early stop")
