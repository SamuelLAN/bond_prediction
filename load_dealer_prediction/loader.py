import os
import numpy as np
import pandas as pd
from lib import utils
from config import path
from load_dealer_prediction.pipelines import load_from_pkl, split_data


class Loader:

    def __init__(self, start_date, end_date, split_bound, min_offering_date):
        self.__start_date = start_date
        self.__end_date = end_date
        self.__split_bound = split_bound
        self.__min_offering_date = min_offering_date

    def load(self):
        self.__d_bond_id_2_trace_for_train, self.__d_bond_id_2_trace_for_test = split_data.run(
            load_from_pkl.run(self.__start_date, self.__end_date, self.__min_offering_date),
            self.__split_bound,
            self.__start_date,
            self.__end_date,
            self.__min_offering_date
        )

    def clustering(self):
        pass


o_loader = Loader('', '', '', '')
o_loader.load()
