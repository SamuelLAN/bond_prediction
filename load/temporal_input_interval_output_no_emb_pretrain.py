import os
import numpy as np
from gensim import corpora
from config import load, date, path
from lib import utils


class Loader:

    def __init__(self, input_days=20, force_refresh=False):
        self.__input_days = input_days
        self.__output_days = input_days // 10
        self.__input_mode = 0

        # get the path of the cache data
        data_subset = load.DATA_SUBSET_FOR_TEMPORAL_INPUT
        subset = os.path.split(data_subset)[1]
        year = load.YEAR_FOR_TEMPORAL_INPUT
        volume_level = load.VOLUME_LEVEL_FOR_TEMPORAL_INPUT
        no_below = load.NO_BELOW_FOR_TEMPORAL_INPUT
        data_index = load.DATA_INDEX_FOR_TEMPORAL_INPUT

        data_all_root_dir = os.path.join(data_subset, year, volume_level)
        all_level = os.path.split(data_all_root_dir)[1]

        self.__data_pkl_path = os.path.join(
            path.PATH_TMP_DIR,
            f'temporal_input_interval_output_for_pretrain_same_volume_{all_level}_{subset}_{year}_{volume_level}_{data_index}_no_below_{no_below}_input_days_{input_days}.pkl')

        if os.path.isfile(self.__data_pkl_path) and not force_refresh:
            self.__train_X, self.__train_y, self.__test_X, self.__test_y, self.dict, self.voc_size = \
                utils.load_pkl(self.__data_pkl_path)

        else:
            print(f'\nStart loading data from {data_all_root_dir} ...')

            train_start_timestamp = utils.date_2_timestamp('2015-01-02')
            train_end_timestamp = utils.date_2_timestamp('2015-10-14', True)

            test_start_timestamp = utils.date_2_timestamp('2015-10-14')
            test_end_timestamp = utils.date_2_timestamp('2015-12-31', True)

            data_all_pkl_path = os.path.join(path.PATH_TMP_DIR,
                                             f'all_doc_list_for_pretrain_{subset}_{year}_{all_level}.pkl')

            if os.path.isfile(data_all_pkl_path):
                train_all_doc_list, test_all_doc_list = utils.load_pkl(data_all_pkl_path)

            else:
                train_all_doc_list = self.__load_dir_all(data_all_root_dir, train_start_timestamp,
                                                         train_end_timestamp,
                                                         'train')
                test_all_doc_list = self.__load_dir_all(data_all_root_dir, test_start_timestamp, test_end_timestamp,
                                                        'test')

                # train_all_doc_list = []
                # test_all_doc_list = []
                # for _volume in os.listdir(data_all_root_dir):
                #     sub_all_root_dir = os.path.join(data_all_root_dir, _volume)
                #     sub_train_all_doc_list = self.__load_dir_all(sub_all_root_dir, train_start_timestamp,
                #                                                  train_end_timestamp,
                #                                                  'train')
                #     sub_test_all_doc_list = self.__load_dir_all(sub_all_root_dir, test_start_timestamp,
                #                                                 test_end_timestamp,
                #                                                 'test')
                #
                #     train_all_doc_list += sub_train_all_doc_list
                #     test_all_doc_list += sub_test_all_doc_list

                utils.write_pkl(data_all_pkl_path, [train_all_doc_list, test_all_doc_list])

            print(f'Finish loading \n\nStart processing data ... ')

            train_all_docs = []
            for v in train_all_doc_list:
                train_all_docs += v
            test_all_docs = []
            for v in test_all_doc_list:
                test_all_docs += v
            del train_all_doc_list
            del test_all_doc_list

            self.dict, self.voc_size = self.__gen_dict(train_all_docs, 150)

            # # generate the dictionary which maps the bond_id to index
            # self.dict, self.voc_size = self.__gen_dict(train_all_doc_list, no_below)

            print(self.voc_size)
            # print(self.voc_size_all)
            print('Finish generating dict\n\nStart converting input output ...')

            # convert doc list to trainable interval summed one-hot vector
            self.__train_X = self.__convert_input(train_all_docs, self.dict, self.voc_size, 'allow_unknown')
            self.__train_y = self.__convert_output(train_all_docs)
            self.__test_X = self.__convert_input(test_all_docs, self.dict, self.voc_size, 'allow_unknown')
            self.__test_y = self.__convert_output(test_all_docs)

            # self.__train_X = 0.
            # self.__test_X = 0.
            # self.__train_y = 0.
            # self.__test_y = 0.
            #
            # for doc_list in train_all_doc_list:
            #     self.__train_X += self.__convert_input(doc_list, self.dict, self.voc_size, 'allow_unknown')
            #     self.__train_y += self.__convert_output(doc_list)
            # self.__train_X /= len(train_all_doc_list)
            #
            # for doc_list in test_all_doc_list:
            #     self.__test_X += self.__convert_input(doc_list, self.dict, self.voc_size, 'allow_unknown')
            #     self.__test_y += self.__convert_output(doc_list)
            # self.__test_X /= len(test_all_doc_list)

            # self.__train_all_X = np.array(
            #     list(map(lambda x: self.__convert_input(x, self.dict_all, self.voc_size_all, 'allow_unknown'),
            #              train_all_doc_list)))
            # self.__train_all_y = np.array(list(map(self.__convert_output, train_all_doc_list)))
            # self.__test_all_X = np.array(
            #     list(map(lambda x: self.__convert_input(x, self.dict_all, self.voc_size_all, 'allow_unknown'),
            #              test_all_doc_list)))
            # self.__test_all_y = np.array(list(map(self.__convert_output, test_all_doc_list)))

            # self.__train_all_X = np.mean(self.__train_all_X, axis=0)
            # self.__train_all_y = np.mean(self.__train_all_y, axis=0)
            # self.__test_all_X = np.mean(self.__test_all_X, axis=0)
            # self.__test_all_y = np.mean(self.__test_all_y, axis=0)

            # # convert doc list to trainable interval summed one-hot vector
            # self.__train_X = self.__convert_input(train_doc_list, self.dict, self.voc_size, 'allow_unknown')
            # self.__train_y = self.__convert_output(train_doc_list)
            # self.__test_X = self.__convert_input(test_doc_list, self.dict, self.voc_size, 'allow_unknown')
            # self.__test_y = self.__convert_output(test_doc_list)

            # self.__train_X = np.vstack([self.__train_X.transpose([2, 0, 1]), self.__train_all_X.transpose([2, 0, 1])])
            # self.__test_X = np.vstack([self.__test_X.transpose([2, 0, 1]), self.__test_all_X.transpose([2, 0, 1])])
            # self.__train_X = self.__train_X.transpose([1, 2, 0])
            # self.__test_X = self.__test_X.transpose([1, 2, 0])

            print('Finish processing ')

            # cache data for faster processing next time
            utils.write_pkl(self.__data_pkl_path, [self.__train_X, self.__train_y, self.__test_X, self.__test_y,
                                                   self.dict, self.voc_size])

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
        cur_timestamp = start_timestamp = utils.date_2_timestamp(start_date)
        end_timestamp = utils.date_2_timestamp(end_date, True)

        # traverse all the date between the start date and the end date, but skip the holidays
        while cur_timestamp < end_timestamp:
            _date = utils.timestamp_2_date(cur_timestamp)
            file_name = f'doc_{_date}.json'

            # if it is holidays, then skip it
            if date.is_holiday(_date):
                pass

            # check if there is any transaction
            elif file_name in date_dict:
                file_path = os.path.join(dir_path, file_name)

                # remove nan in doc
                tmp_doc = list(map(lambda x: x if isinstance(x, str) else '', utils.load_json(file_path)))
                while '' in tmp_doc:
                    tmp_doc.remove('')

                data.append(tmp_doc)

            # if no transaction happens in that date
            else:
                data.append([])

            # move to the next day
            cur_timestamp += 86400

        return data, start_timestamp, end_timestamp

    def __load_dir_all(self, dir_path, start_timestamp, end_timestamp, mode='train'):
        data = []
        for data_index in os.listdir(dir_path):
            tmp_doc_list = self.__load_dir_one(os.path.join(dir_path, data_index, mode), start_timestamp, end_timestamp)
            data.append(tmp_doc_list)
        return data

    @staticmethod
    def __load_dir_one(dir_path, cur_timestamp, end_timestamp):
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

        # traverse all the date between the start date and the end date, but skip the holidays
        while cur_timestamp < end_timestamp:
            _date = utils.timestamp_2_date(cur_timestamp)
            file_name = f'doc_{_date}.json'

            # if it is holidays, then skip it
            if date.is_holiday(_date):
                pass

            # check if there is any transaction
            elif file_name in date_dict:
                file_path = os.path.join(dir_path, file_name)

                # remove nan in doc
                tmp_doc = list(map(lambda x: x if isinstance(x, str) else '', utils.load_json(file_path)))
                while '' in tmp_doc:
                    tmp_doc.remove('')

                data.append(tmp_doc)

            # if no transaction happens in that date
            else:
                data.append([])

            # move to the next day
            cur_timestamp += 86400

        return data

    @staticmethod
    def __gen_dict(doc_list, no_below):
        """ generate the dictionary which maps the bond_id to index """
        dictionary = corpora.Dictionary(doc_list)
        dictionary.filter_extremes(no_below=no_below)

        # dictionary size plus one unknown
        voc_size = len(dictionary) + 1

        return dictionary, voc_size

    def __convert_input(self, doc_list, _dict, voc_size, mode='allow_unknown'):
        dates = list(map(lambda x: _dict.doc2idx(x), doc_list))
        if mode != 'allow_unknown':
            dates = list(map(lambda x: [v for v in x if v != -1], doc_list))
        dates = list(map(lambda x: self.__2_sum_one_hot(x, voc_size), dates))

        len_data = len(dates) - (self.__input_days + self.__output_days) + 1
        return np.array([dates[i: i + self.__input_days] for i in range(len_data)])

    def __convert_output(self, doc_list):
        """ convert the bond doc list to trainable interval summed one-hot vector """
        dates = list(map(lambda x: self.dict.doc2idx(x), doc_list))
        dates = list(map(lambda x: self.__2_sum_one_hot(x, self.voc_size), dates))

        len_data = len(dates) - (self.__input_days + self.__output_days) + 1
        return np.array([
            self.__process_temporal_output(dates[i + self.__input_days: i + self.__input_days + self.__output_days])
            for i in range(len_data)
        ])

    def __2_one_hot(self, x, voc_size):
        return np.eye(voc_size)[x]

    def __2_sum_one_hot(self, x, voc_size, max_0_val=None):
        l = np.sum(self.__2_one_hot(x, voc_size), axis=0)
        if max_0_val:
            l[l > 0] = max_0_val
        return l

    @staticmethod
    def __process_temporal_output(y_list):
        """ process temporal output to the output format that the model needs """
        y = np.sum(y_list, axis=0)
        y[y > 0] = 1
        return y

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

    def __gen_topics_mask(self):
        topics = utils.load_json(path.TOPIC_BONDS_JSON)
        topics = self.dict.doc2idx(topics)
        while -1 in topics:
            topics.remove(-1)

        self.__topic_mask = self.__2_sum_one_hot(topics, self.voc_size)
        self.__topic_mask[self.__topic_mask > 0] = 1

    def topic_mask(self):
        return self.__topic_mask

    def train(self):
        """ return train_X, train_Y """
        return self.__train_X, np.array(self.__train_y, np.int32)

    def test(self):
        """ return test_X, test_Y """
        return self.__test_X, np.array(self.__test_y, np.int32)

# o_load = Loader(force_refresh=True)
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
