import os
import numpy as np
from gensim import corpora
from config import load, date, path
from lib import utils


class Loader:

    def __init__(self, input_days=20):
        self.__input_days = input_days
        self.__output_days = input_days // 10

        # get the path of the cache data
        data_subset = load.DATA_SUBSET_FOR_DEALER_PRED
        subset = os.path.split(data_subset)[1]
        year = load.YEAR_FOR_DEALER_PRED
        volume_level = load.VOLUME_LEVEL_FOR_DEALER_PRED
        no_below = load.NO_BELOW_FOR_DEALER_PRED
        data_index = load.DATA_INDEX_FOR_DEALER_PRED

        self.__data_pkl_path = os.path.join(
            path.PATH_TMP_DIR,
            f'interval_input_output_{subset}_{year}_{volume_level}_{data_index}_no_below_{no_below}_input_days_{input_days}.pkl')

        if os.path.isfile(self.__data_pkl_path):
            self.__train_data, self.__test_data, self.dict, self.voc_size = utils.load_pkl(self.__data_pkl_path)

        else:

            data_root_dir = os.path.join(data_subset, year, volume_level, data_index)

            print(f'\nStart loading data from {data_root_dir} ...')

            # load doc list
            train_doc_list = self.__load_dir(os.path.join(data_root_dir, 'train'))
            test_doc_list = self.__load_dir(os.path.join(data_root_dir, 'test'))

            print(f'Finish loading \n\nStart processing data ... ')

            # generate the dictionary which maps the bond_id to index
            self.dict, self.voc_size = self.__gen_dict(train_doc_list)

            # convert doc list to trainable interval summed one-hot vector
            self.__train_data = self.__convert(train_doc_list)
            self.__test_data = self.__convert(test_doc_list)

            print('Finish processing ')

            # cache data for faster processing next time
            utils.write_pkl(self.__data_pkl_path, [self.__train_data, self.__test_data, self.dict, self.voc_size])

        self.__gen_topics_mask()

        self.__statistic()

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

    def __convert(self, doc_list):
        """ convert the bond doc list to trainable interval summed one-hot vector """
        dates = list(map(lambda x: self.dict.doc2idx(x), doc_list))
        dates = list(map(self.__2_sum_one_hot, dates))
        return self.__pick_input_output(dates, self.__input_days, self.__output_days)

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
        train_label_cardinality = np.sum(self.__train_data[:, 1]) / len(self.__train_data[:, 1])
        test_label_cardinality = np.sum(self.__test_data[:, 1]) / len(self.__test_data[:, 1])

        train_label_density = np.mean(self.__train_data[:, 1])
        test_label_density = np.mean(self.__test_data[:, 1])

        # add the statistic to log
        load.LOG['train_label_cardinality'] = train_label_cardinality
        load.LOG['test_label_cardinality'] = test_label_cardinality
        load.LOG['train_label_density'] = train_label_density
        load.LOG['test_label_density'] = test_label_density
        load.LOG['voc_size'] = self.voc_size

        print(f'train_label_cardinality: {train_label_cardinality}')
        print(f'test_label_cardinality: {test_label_cardinality}')
        print(f'train_label_density: {train_label_density}')
        print(f'test_label_density: {test_label_density}')
        print(f'voc_size: {self.voc_size}')

    def __gen_topics_mask(self):
        topics = utils.load_json(path.TOPIC_BONDS_JSON)
        topics = self.dict.doc2idx(topics)
        while -1 in topics:
            topics.remove(-1)

        self.__topic_mask = self.__2_sum_one_hot(topics)
        self.__topic_mask[self.__topic_mask > 0] = 1

    def topic_mask(self):
        return self.__topic_mask

    def train(self):
        """ return train_X, train_Y """
        return self.__train_data[:, 0], np.array(self.__train_data[:, 1], np.int32)

    def test(self):
        """ return test_X, test_Y """
        return self.__test_data[:, 0], np.array(self.__test_data[:, 1], np.int32)


# o_load = Loader()
# train_x, train_y = o_load.train()
# test_x, test_y = o_load.test()
#
# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)
#
# topic_mask = o_load.topic_mask()
# print('\n********************************')
# print(topic_mask)
# print(topic_mask.shape)
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
