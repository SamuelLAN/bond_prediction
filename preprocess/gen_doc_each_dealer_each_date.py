import os
import json
import time
import numpy as np
from config import path
from lib import utils


def generate_doc_list(dir_path, new_dir_path, id_index=0):
    """
    Generate the doc for each dealer in each date
        there is only one feature (the bond itself) and there is no filter
    """

    print('\nstart generating doc list from %s to %s ...' % (dir_path, new_dir_path))

    file_list = os.listdir(dir_path)
    len_file_list = len(file_list)

    for i, file_name in enumerate(file_list):
        if i % 5 == 0:
            progress = float(i + 1) / len_file_list * 100.
            print('\rprogress: %.2f%% ' % progress, end='')

        file_name_prefix = os.path.splitext(file_name)[0]

        # load bonds data
        file_path = os.path.join(dir_path, file_name)
        data = utils.load_json(file_path)

        # to fill the date gaps (because there are dates that no bond transactions happen)
        d_date = {}

        # parse the date to timestamp
        start_date = str(data[0][-2])[:-9] + ' 00:00:00'
        end_date = str(data[-1][-2])[:-9] + ' 23:00:00'
        cur_time_stamp = time.mktime(time.strptime(start_date, '%Y-%m-%d %H:%M:%S'))
        end_time_stamp = time.mktime(time.strptime(end_date, '%Y-%m-%d %H:%M:%S'))

        # traverse the dates and fill the gaps
        while cur_time_stamp <= end_time_stamp:
            date_time = time.localtime(cur_time_stamp)
            date = time.strftime('%Y-%m-%d', date_time)
            cur_time_stamp += 86400
            d_date[date] = []

        # fill the bonds to the d_date dict
        for v in data:
            _date_time = str(v[-2])
            _date = _date_time[:-9]

            if id_index == 0:
                _id = v[id_index]
                val = [v[0], v[-1], _date_time]
            else:
                val = [v[1], v[2], v[-1], _date_time]

            d_date[_date].append(val)

        # convert the d_date (dict) to list
        l_date = list(d_date.items())
        l_date.sort(key=lambda x: x[0])

        while l_date[-1][0][:4] != l_date[0][0][:4]:
            del l_date[-1]

        dealer_dir = os.path.join(new_dir_path, file_name_prefix)
        if not os.path.isdir(dealer_dir):
            os.mkdir(dealer_dir)

        for _date, _data_in_a_day in l_date:
            if not _data_in_a_day:
                continue

            _data_in_a_day.sort(key=lambda x: x[-1])

            _data_in_a_day = list(map(lambda x: x[:-1], _data_in_a_day))
            utils.write_json(os.path.join(dealer_dir, f'doc_{_date}.json'), _data_in_a_day)

    print('\nfinish generating ')


for dir_name in os.listdir(path.PREDICTION_DIR):
    dir_path = os.path.join(path.PREDICTION_DIR, dir_name)
    new_dir_path = os.path.join(path.PREDICTION_DATE_DIR, dir_name)

    if dir_name[:4] == 'bond':
        idx = 0
    else:
        idx = 1

    if not os.path.isdir(new_dir_path):
        os.mkdir(new_dir_path)

    for year in os.listdir(dir_path):
        year_path = os.path.join(dir_path, year)
        new_year_path = os.path.join(new_dir_path, year)

        if not os.path.isdir(new_year_path):
            os.mkdir(new_year_path)

        generate_doc_list(year_path, new_year_path, idx)


print('\ndone')
