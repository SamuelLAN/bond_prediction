import os
import random
import numpy as np
from gensim import corpora
from config import load, path
from lib.utils import load_json, write_json, load_pkl, write_pkl

RANDOM_STATE = 42


class Loader:
    def __init__(self, window=1, mode='cbow'):
        self.__window = window
        self.__mode = mode
        self.__emb_pkl_path = os.path.join(path.PATH_TMP_DIR,
                                           f'emb_data_w_{window}_no_below_{load.NO_BELOW_BONDS_FOR_EMB}.pkl')
        self.__test_emb_pkl_path = os.path.join(path.PATH_TMP_DIR,
                                                f'emb_test_data_w_{window}_no_below_{load.NO_BELOW_BONDS_FOR_EMB}.pkl')

        self.__load_train()
        self.__load_test()

        self.__train_X, self.__train_y = self.__shuffle(self.__train_X, self.__train_y)
        self.__test_X, self.__test_y = self.__shuffle(self.__test_X, self.__test_y)

    @staticmethod
    def __shuffle(X, y):
        random_indices = list(range(len(X)))
        random.seed(RANDOM_STATE)
        random.shuffle(random_indices)
        return X[random_indices], y[random_indices]

    def __load_train(self):
        """ load train data """
        print('\nStart loading train data')

        if os.path.isfile(self.__emb_pkl_path):
            self.__train_X, self.__train_y, self.dict, self.voc_size = load_pkl(self.__emb_pkl_path)

        else:
            print('loading doc list ...')

            # load the doc_list
            emb_json_path = os.path.join(path.PATH_TMP_DIR, 'emb_data.json')
            if os.path.isfile(emb_json_path):
                docs = load_json(emb_json_path)
            else:
                path_list = self.__get_path_list()
                docs = self.__load_docs(path_list, emb_json_path)

            print('generating dictionary ...')

            # generate the dictionary which maps the bond_id to index
            self.dict, self.voc_size = self.__gen_dict(docs)

            print('converting docs to trainable data format ...')

            # convert the doc list to trainable data format
            self.__train_X, self.__train_y = self.__convert(docs, self.__emb_pkl_path)

        print('Finish loading train data')

    def __load_test(self):
        """ load test data """
        print('\nStart loading test data ...')

        if os.path.isfile(self.__test_emb_pkl_path):
            self.__test_X, self.__test_y, _, _ = load_pkl(self.__test_emb_pkl_path)

        else:
            print('loading test doc list ...')

            # load the doc_list
            emb_json_path = os.path.join(path.PATH_TMP_DIR, 'emb_test_data.json')
            if os.path.isfile(emb_json_path):
                docs = load_json(emb_json_path)
            else:
                path_list = self.__get_path_list('test')
                docs = self.__load_docs(path_list, emb_json_path)

            print('converting test docs to trainable test data format ...')

            # convert the doc list to trainable data format
            self.__test_X, self.__test_y = self.__convert(docs, self.__test_emb_pkl_path)

        print('Finish loading test data')

    @staticmethod
    def __get_path_list(mode='train'):
        """ traverse the data dirs to get the path list of all data """
        path_list = []
        for dir_path in load.DATA_DIR_LIST:

            for dealer_dir_name in os.listdir(dir_path):
                dealer_dir = os.path.join(dir_path, dealer_dir_name, mode)

                for _date in os.listdir(dealer_dir):
                    file_path = os.path.join(dealer_dir, _date)
                    path_list.append(file_path)

        return path_list

    @staticmethod
    def __load_docs(path_list, emb_json_path):
        """ load all the data from the path list """
        docs = []
        length = len(path_list)

        # traverse the path list to load all the data
        for i, _path in enumerate(path_list):
            # show progress
            if i % 5 == 0:
                progress = float(i + 1) / length * 100.
                print('\rprogress: %.2f%% ' % progress, end='')

            # remove nan in doc
            tmp_doc = list(map(lambda x: x if isinstance(x, str) else '', load_json(_path)))
            while '' in tmp_doc:
                tmp_doc.remove('')

            docs.append(tmp_doc)

        # cache data for faster processing next time
        write_json(emb_json_path, docs)
        return docs

    @staticmethod
    def __gen_dict(doc_list):
        """ generate the dictionary which maps the bond_id to index """
        dictionary = corpora.Dictionary(doc_list)
        dictionary.filter_extremes(no_below=load.NO_BELOW_BONDS_FOR_EMB)

        # dictionary size plus one unknown
        voc_size = len(dictionary) + 1

        return dictionary, voc_size

    def __convert(self, docs, pkl_path):
        """ convert the doc list to trainable data format """
        docs = list(map(
            lambda x: list((np.array(self.dict.doc2idx(x)) + self.voc_size) % self.voc_size) if x else x, docs))
        X = []
        y = []

        for i, doc in enumerate(docs):
            len_doc = len(doc)
            len_data = len_doc - (2 * self.__window + 1) + 1
            if len_data <= 0:
                continue

            X += [doc[j: j + self.__window] + doc[j + self.__window + 1: j + 2 * self.__window + 1]
                  for j in range(len_data)]
            y += [doc[j + self.__window] for j in range(len_data)]

        X = np.array(X)
        y = np.array(y)

        X, y = self.__shuffle(X, y)

        # cache data for faster processing next time
        write_pkl(pkl_path, [X, y, self.dict, self.voc_size])

        return X, y

    def train(self):
        if self.__mode == 'cbow':
            return self.__train_X, self.__train_y
        else:
            return self.__train_y, self.__train_X

    def test(self):
        if self.__mode == 'cbow':
            return self.__test_X, self.__test_y
        else:
            return self.__test_y, self.__test_X

# o_loader = Loader(window=3)
# train_X, train_y = o_loader.train()
# test_X, test_y = o_loader.test()
#
# print(f'\nvoc_size: {o_loader.voc_size}\ndone')
# print(train_X.shape)
# print(train_y.shape)
# print(test_X.shape)
# print(test_y.shape)
#
# for i in range(10000):
#     print('\n-------------------')
#     print(train_X[i])
#     print(train_y[i])
#     print('')
