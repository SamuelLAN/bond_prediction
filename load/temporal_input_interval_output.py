import os
import warnings
import numpy as np
from gensim import corpora

warnings.filterwarnings('ignore')

import tensorflow as tf

# tf.enable_eager_execution()

from config import load, date, path
from lib import utils
from models.word2vec import Model as Word2vec

keras = tf.keras
layers = keras.layers


class Loader:

    def __init__(self, input_days=20):
        self.__input_days = input_days
        self.__output_days = input_days // 10

        # get the path of the cache data
        data_subset = load.DATA_SUBSET_FOR_TEMPORAL_INPUT
        subset = os.path.split(data_subset)[1]
        year = load.YEAR_FOR_TEMPORAL_INPUT
        volume_level = load.VOLUME_LEVEL_FOR_TEMPORAL_INPUT
        no_below = load.NO_BELOW_FOR_TEMPORAL_INPUT
        data_index = load.DATA_INDEX_FOR_TEMPORAL_INPUT

        self.__data_pkl_path = os.path.join(
            path.PATH_TMP_DIR,
            f'temporal_input_interval_output_emb_{subset}_{year}_{volume_level}_{data_index}_no_below_{no_below}_input_days_{input_days}.pkl')

        self.__test_emb_pkl_path = os.path.join(path.PATH_TMP_DIR,
                                                f'emb_test_data_w_3_no_below_1000.pkl')

        if os.path.isfile(self.__data_pkl_path):
            self.__train_X, self.__train_y, self.__test_X, self.__test_y, \
            self.emb_dict, self.emb_voc_size, self.dict, self.voc_size = \
                utils.load_pkl(self.__data_pkl_path)

        else:
            print('\nStart loading embedding model ... ')
            self.__load_emb_model()
            print('Finish loading embedding model ')

            _, _, self.emb_dict, self.emb_voc_size = utils.load_pkl(self.__test_emb_pkl_path)

            data_root_dir = os.path.join(data_subset, year, volume_level, data_index)

            print(f'\nStart loading data from {data_root_dir} ...')

            # load doc list
            train_doc_list = self.__load_dir(os.path.join(data_root_dir, 'train'))
            test_doc_list = self.__load_dir(os.path.join(data_root_dir, 'test'))

            print(f'Finish loading \n\nStart generating dict for output ... ')

            # generate the dictionary which maps the bond_id to index
            self.dict, self.voc_size = self.__gen_dict(train_doc_list)

            print('Finish generating\n\nStart converting data ...')

            # convert doc list to trainable interval summed one-hot vector
            self.__train_X = self.__convert_input(train_doc_list)
            self.__train_y = self.__convert_output(train_doc_list)
            self.__test_X = self.__convert_input(test_doc_list)
            self.__test_y = self.__convert_output(test_doc_list)

            print('Finish processing ')

            # cache data for faster processing next time
            utils.write_pkl(self.__data_pkl_path, [self.__train_X, self.__train_y, self.__test_X, self.__test_y,
                                                   self.emb_dict, self.emb_voc_size, self.dict, self.voc_size])

        self.__statistic()

    def __load_emb_model(self):
        emb_model = Word2vec('2020_01_02_23_55_42', 'word2vec')
        emb_model.load_model()
        emb_matrix = emb_model.emb_matrix().numpy()

        emb_layer = layers.Embedding(Word2vec.VOC_SIZE, Word2vec.EMB_SIZE,
                                     embeddings_initializer=tf.constant_initializer(emb_matrix))
        self.__emb_model = keras.Sequential([
            emb_layer,
            layers.Lambda(lambda x: tf.reduce_sum(x, axis=1)),
        ])

    def __convert_one_day_bonds_2_a_emb(self, bonds_in_a_day):
        return np.sum(self.__emb_model.predict(np.array(bonds_in_a_day)), axis=0)

    def __convert_one_input_sample_2_embs(self, bonds_list):
        return list(map(self.__convert_one_day_bonds_2_a_emb, bonds_list))

    @staticmethod
    def __load_dir(dir_path):
        """
        Load all the data in "dir_path", and complement the data in the dates that no transaction happened
        :return
            data: (list)
            e.g. [ # include transactions happen in many days
                ['bond_a', 'bond_b', ...], # represent transaction happen in one day
                ['bond_a', 'bond_b', ...],
                ...
            ]
        """
        data = []

        # load the date list
        date_list = os.listdir(dir_path)
        date_list.sort()

        # generate a date dict so that we can check whether there is transaction happens in that date
        date_dict = utils.list_2_dict(date_list)

        # find out the start and end date of all the transactions
        start_date = date_list[0][len('doc_'): -len('.json')]
        end_date = date_list[-1][len('doc_'): -len('.json')]

        # covert the date to timestamp
        cur_timestamp = utils.date_2_timestamp(start_date)
        end_timestamp = utils.date_2_timestamp(end_date) + 86000

        # traverse all the date between the start date and the end date, but skip the holidays
        while cur_timestamp < end_timestamp:
            _date = utils.timestamp_2_date(cur_timestamp)
            file_name = f'doc_{_date}.json'

            # check if there is any transaction
            if file_name in date_dict:
                file_path = os.path.join(dir_path, file_name)

                # remove nan in doc
                tmp_doc = list(map(lambda x: x if isinstance(x, str) else '', utils.load_json(file_path)))
                while '' in tmp_doc:
                    tmp_doc.remove('')

                data.append(tmp_doc)

            # if it is holidays, then skip it
            elif date.is_holiday(_date):
                pass

            # if no transaction happens in that date
            else:
                data.append([])

            # move to the next day
            cur_timestamp += 86400

        return data

    @staticmethod
    def __gen_dict(doc_list):
        """ generate the dictionary which maps the bond_id to index """
        dictionary = corpora.Dictionary(doc_list)
        dictionary.filter_extremes(no_below=load.NO_BELOW)

        # dictionary size plus one unknown
        voc_size = len(dictionary) + 1

        return dictionary, voc_size

    def __convert_input(self, doc_list):
        dates = list(map(
            lambda x: (np.array(self.emb_dict.doc2idx(x)) + self.emb_voc_size) % self.emb_voc_size if x else x,
            doc_list))

        len_data = len(dates) - (self.__input_days + self.__output_days) + 1
        return np.array([
            self.__convert_one_input_sample_2_embs(dates[i: i + self.__input_days])
            for i in range(len_data)
        ])

    def __convert_output(self, doc_list):
        """ convert the bond doc list to trainable interval summed one-hot vector """
        dates = list(map(lambda x: self.dict.doc2idx(x), doc_list))
        dates = list(map(self.__2_sum_one_hot, dates))

        len_data = len(dates) - (self.__input_days + self.__output_days) + 1
        return np.array([
            self.__process_temporal_output(dates[i + self.__input_days: i + self.__input_days + self.__output_days])
            for i in range(len_data)
        ])

    def __2_one_hot(self, x):
        return np.eye(self.voc_size)[x]

    def __2_sum_one_hot(self, x):
        return np.sum(self.__2_one_hot(x), axis=0)

    @staticmethod
    def __process_temporal_input(x_list):
        """ process temporal input to the input format that the model needs """
        return np.sum(x_list, axis=0)

    @staticmethod
    def __process_temporal_output(y_list):
        """ process temporal output to the output format that the model needs """
        y = np.sum(y_list, axis=0)
        y[y > 0] = 1
        return y

    def __pick_input_output(self, dates, input_days, output_days):
        """ get the temporal input and output data, and then sum it to one-hot vector respectively """
        len_data = len(dates) - (input_days + output_days) + 1
        return np.array([[
            self.__process_temporal_input(dates[i: i + input_days]),  # input
            self.__process_temporal_output(dates[i + input_days: i + input_days + output_days]),  # output
        ] for i in range(len_data)])

    def __statistic(self):
        train_label_cardinality = np.sum(self.__train_y) / len(self.__train_y)
        test_label_cardinality = np.sum(self.__test_y) / len(self.__test_y)

        train_label_density = np.mean(self.__train_y)
        test_label_density = np.mean(self.__test_y)

        # add the statistic to log
        load.LOG['train_label_cardinality'] = train_label_cardinality
        load.LOG['test_label_cardinality'] = test_label_cardinality
        load.LOG['train_label_density'] = train_label_density
        load.LOG['test_label_density'] = test_label_density
        load.LOG['voc_size'] = self.voc_size

        print('\n---------------------------')
        print(f'train_label_cardinality: {train_label_cardinality}')
        print(f'test_label_cardinality: {test_label_cardinality}')
        print(f'train_label_density: {train_label_density}')
        print(f'test_label_density: {test_label_density}')
        print(f'voc_size: {self.voc_size}')

    def train(self):
        """ return train_X, train_Y """
        return self.__train_X, self.__train_y

    def test(self):
        """ return test_X, test_Y """
        return self.__test_X, self.__test_y


# o_load = Loader()
# train_x, train_y = o_load.train()
# test_x, test_y = o_load.test()
#
# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)
#
# print('\n-------------------------------------')
# print(train_x[12])
# print(train_y[12])
#
# print('-------------------------------------')
# print(train_x[23])
# print(train_y[23])
#
# print('-------------------------------------')
# print(train_x[65])
# print(train_y[65])
