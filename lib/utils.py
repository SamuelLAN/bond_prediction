import os
import time
import json
import chardet
from six.moves import cPickle as pickle


def date_2_timestamp(_date, close_night=False):
    _time = ' 00:00:00' if not close_night else ' 23:00:00'
    return time.mktime(time.strptime(_date + _time, '%Y-%m-%d %H:%M:%S'))


def date_2_weekday(_date):
    return time.strftime('%w', time.strptime(_date, '%Y-%m-%d'))


def timestamp_2_date(timestamp):
    return time.strftime('%Y-%m-%d', time.localtime(timestamp))


def list_2_dict(_list):
    d = {}
    for v in _list:
        d[v] = True
    return d


def load_json(_path):
    with open(_path, 'r') as f:
        return json.load(f)


def write_json(_path, data):
    with open(_path, 'w') as f:
        json.dump(data, f)


def load_pkl(_path):
    with open(_path, 'rb') as f:
        return pickle.load(f)


def write_pkl(_path, data):
    with open(_path, 'wb') as f:
        pickle.dump(data, f)


def decode_2_utf8(string):
    if isinstance(string, str):
        return string
    if isinstance(string, int) or isinstance(string, float):
        return str(string)
    if not isinstance(string, bytes):
        return string

    try:
        return string.decode('utf-8')
    except:
        encoding = chardet.detect(string)['encoding']
        if encoding:
            try:
                return string.decode(encoding)
            except:
                pass
        return string


def output_and_log(file_path, output, headline=''):
    """ Display the output to the console and save it to the log file. """
    # show to the console
    print(output)
    # save to the log file
    content = headline + output if not os.path.exists(file_path) else output
    with open(file_path, 'ab') as f:
        f.write(str(content + '\n').encode('utf-8'))
