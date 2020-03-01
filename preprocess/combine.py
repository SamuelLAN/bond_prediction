import os
import json
import numpy as np
from config import path
from six.moves import cPickle as pickle

d_dealer = {}
d_dealer_buy_client = {}
d_dealer_sell_client = {}
d_dealer_buy_dealer = {}
d_dealer_sell_dealer = {}
d_dealer_clients = {}
d_dealer_dealers = {}

d_bond = {}
d_bond_buy_client = {}
d_bond_sell_client = {}
d_bond_clients = {}
d_bond_dealers = {}


def __save(_dict, _dir_path, prefix):
    print(f'\nstart saving data to {_dir_path} with prefix {prefix} ... ')

    if not os.path.isdir(_dir_path):
        os.mkdir(_dir_path)

    len_dict = len(_dict)
    count = 0
    for _index, _list in _dict.items():
        if count % 20 == 0:
            progress = float(count + 1) / len_dict * 100.
            print('\rprogress: %.2f%% ' % progress, end='')
        count += 1

        _file_name = f'{prefix}_{_index}.json'
        _file_path = os.path.join(_dir_path, _file_name)
        _list.sort(key=lambda x: x[-2])

        with open(_file_path, 'w') as f:
            json.dump(_list, f)

        # print(f'save {_file_name} successful ')


for file_name in os.listdir(path.TRACE_DIR):
    year = os.path.splitext(file_name)[0].split('_')[-1]
    file_path = os.path.join(path.TRACE_DIR, file_name)

    if year != '2015':
        continue

    print('loading %s ... ' % file_path)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f'finish loading {file_path}')

    print('\nstart converting data ...')
    data = np.array(data)

    # [ bond_id, dealer_index, Contra_Party_Index, rating_mr_numeric, price, volume, rpt_side_cd, date ]
    data = list(map(lambda x: [x[0], x[10], x[11], x[14], x[4], x[3], x[5], str(x[9])], data))
    print('finish converting ')

    print('\nadding data to dict ...')
    length = len(data)
    for i, v in enumerate(data):
        if i % 20 == 0:
            progress = float(i + 1) / length * 100.
            print('\rprogress: %.2f%% ' % progress, end='')

        bond_id = v[0]
        rpt_dealer_index = int(v[1])
        contra_party_index = int(v[2])
        rpt_side_cd = v[-2]
        volume = float(v[-3]) * 1000

        # buy from clients
        if rpt_dealer_index == 0:
            dealer_index = contra_party_index
            value = [bond_id, rpt_dealer_index, v[-1], 'bfc']

            if rpt_side_cd != 'C':
                print(f'rpt_side_cd != C {v}')

            # add all
            if dealer_index not in d_dealer:
                d_dealer[dealer_index] = []
            d_dealer[dealer_index].append(value)

            # add to clients
            if dealer_index not in d_dealer_clients:
                d_dealer_clients[dealer_index] = []
            d_dealer_clients[dealer_index].append(value)

            # add buy from clients
            if dealer_index not in d_dealer_buy_client:
                d_dealer_buy_client[dealer_index] = []
            d_dealer_buy_client[dealer_index].append(value)

            value = [bond_id, contra_party_index, rpt_dealer_index, v[-1], 'bfc']

            # add all
            if bond_id not in d_bond:
                d_bond[bond_id] = []
            d_bond[bond_id].append(value)

            # add to clients
            if bond_id not in d_bond_clients:
                d_bond_clients[bond_id] = []
            d_bond_clients[bond_id].append(value)

            # add buy from clients
            if bond_id not in d_bond_buy_client:
                d_bond_buy_client[bond_id] = []
            d_bond_buy_client[bond_id].append(value)

        else:
            # sell to clients
            if contra_party_index == 99999:
                dealer_index = rpt_dealer_index
                value = [bond_id, contra_party_index, v[-1], 'stc']

                if rpt_side_cd != 'C':
                    print(f'rpt_side_cd != C {v}')

                # add all
                if dealer_index not in d_dealer:
                    d_dealer[dealer_index] = []
                d_dealer[dealer_index].append(value)

                # add to clients
                if dealer_index not in d_dealer_clients:
                    d_dealer_clients[dealer_index] = []
                d_dealer_clients[dealer_index].append(value)

                # add sell to clients
                if dealer_index not in d_dealer_sell_client:
                    d_dealer_sell_client[dealer_index] = []
                d_dealer_sell_client[dealer_index].append(value)

                value = [bond_id, rpt_dealer_index, contra_party_index, v[-1], 'stc']

                # add all
                if bond_id not in d_bond:
                    d_bond[bond_id] = []
                d_bond[bond_id].append(value)

                # add to clients
                if bond_id not in d_bond_clients:
                    d_bond_clients[bond_id] = []
                d_bond_clients[bond_id].append(value)

                # add buy from clients
                if bond_id not in d_bond_sell_client:
                    d_bond_sell_client[bond_id] = []
                d_bond_sell_client[bond_id].append(value)

            else:

                if rpt_side_cd != 'D':
                    print(f'rpt_side_cd != D {v}')

                # buy from dealers
                dealer_index = contra_party_index
                value = [bond_id, rpt_dealer_index, v[-1], 'bfd']

                # add all
                if dealer_index not in d_dealer:
                    d_dealer[dealer_index] = []
                d_dealer[dealer_index].append(value)

                # add to dealers
                if dealer_index not in d_dealer_dealers:
                    d_dealer_dealers[dealer_index] = []
                d_dealer_dealers[dealer_index].append(value)

                # add buy from dealers
                if dealer_index not in d_dealer_buy_dealer:
                    d_dealer_buy_dealer[dealer_index] = []
                d_dealer_buy_dealer[dealer_index].append(value)

                # sell to dealers
                dealer_index = rpt_dealer_index
                value = [bond_id, contra_party_index, v[-1], 'std']

                # add all
                if dealer_index not in d_dealer:
                    d_dealer[dealer_index] = []
                d_dealer[dealer_index].append(value)

                # add to dealers
                if dealer_index not in d_dealer_dealers:
                    d_dealer_dealers[dealer_index] = []
                d_dealer_dealers[dealer_index].append(value)

                # add sell to dealers
                if dealer_index not in d_dealer_sell_dealer:
                    d_dealer_sell_dealer[dealer_index] = []
                d_dealer_sell_dealer[dealer_index].append(value)

                value = [bond_id, rpt_dealer_index, contra_party_index, v[-1], 'dtd']

                # add all
                if bond_id not in d_bond:
                    d_bond[bond_id] = []
                d_bond[bond_id].append(value)

                # add to clients
                if bond_id not in d_bond_dealers:
                    d_bond_dealers[bond_id] = []
                d_bond_dealers[bond_id].append(value)

    del data

    print('finish adding ')

    dir_path = os.path.join(path.PREDICTION_BONDS_BY_DEALER_DIR, year)
    __save(d_dealer, dir_path, 'dealer')
    d_dealer = {}

    dir_path = os.path.join(path.PREDICTION_BONDS_BY_DEALER_BUY_CLIENT, year)
    __save(d_dealer_buy_client, dir_path, 'dealer')
    d_dealer_buy_client = {}

    dir_path = os.path.join(path.PREDICTION_BONDS_BY_DEALER_SELL_CLIENT, year)
    __save(d_dealer_sell_client, dir_path, 'dealer')
    d_dealer_sell_client = {}

    dir_path = os.path.join(path.PREDICTION_BONDS_BY_DEALER_BUY_DEALER, year)
    __save(d_dealer_buy_dealer, dir_path, 'dealer')
    d_dealer_buy_dealer = {}

    dir_path = os.path.join(path.PREDICTION_BONDS_BY_DEALER_SELL_DEALER, year)
    __save(d_dealer_sell_dealer, dir_path, 'dealer')
    d_dealer_sell_dealer = {}

    dir_path = os.path.join(path.PREDICTION_BONDS_BY_DEALER_CLIENTS, year)
    __save(d_dealer_clients, dir_path, 'dealer')
    d_dealer_clients = {}

    dir_path = os.path.join(path.PREDICTION_BONDS_BY_DEALER_DEALERS, year)
    __save(d_dealer_dealers, dir_path, 'dealer')
    d_dealer_dealers = {}

    dir_path = os.path.join(path.PREDICTION_DEALERS_BY_BOND_DIR, year)
    __save(d_bond, dir_path, 'bond')
    d_bond = {}

    dir_path = os.path.join(path.PREDICTION_DEALERS_BY_BOND_BUY_CLIENT, year)
    __save(d_bond_buy_client, dir_path, 'bond')
    d_bond_buy_client = {}

    dir_path = os.path.join(path.PREDICTION_DEALERS_BY_BOND_SELL_CLIENT, year)
    __save(d_bond_sell_client, dir_path, 'bond')
    d_bond_sell_client = {}

    dir_path = os.path.join(path.PREDICTION_DEALERS_BY_BOND_CLIENTS, year)
    __save(d_bond_clients, dir_path, 'bond')
    d_bond_clients = {}

    dir_path = os.path.join(path.PREDICTION_DEALERS_BY_BOND_DEALERS, year)
    __save(d_bond_dealers, dir_path, 'bond')
    d_bond_dealers = {}

print('\ndone')

"""

0	         BOND_SYM_ID	              ABN.GG	ABN.GG
1	            CUSIP_ID	           00077TAA2	00077TAA2
2	       SCRTY_TYPE_CD	                   C	C
3	        ENTRD_VOL_QT	              8000.0	4000.0
4	             RPTD_PR	             109.607	90.0
5	         RPT_SIDE_CD	                   C	C
6	                Year	              2008.0	2008.0
7	       document_date	          2008-04-22	2008-10-01
8	      TRD_EXCTN_DTTM	 2008-04-22 15:59:00	2008-10-01 10:55:00
9	        TRD_RPT_DTTM	 2008-04-22 15:59:28	2008-10-01 11:06:00
10	 Report_Dealer_Index	                 151	0
11	  Contra_Party_Index	               99999	151
12	              TRC_ST	                   T	T
13	           RATING_MR	                 Aa3	Aa3
14	   RATING_MR_Numeric	                 4.0	4.0
15	       OFFERING_DATE	 1993-05-20 00:00:00	1993-05-20 00:00:00
16	            ISSUE_ID	                 5.0	5.0
17	            MATURITY	 2023-05-15 00:00:00	2023-05-15 00:00:00
18	        OFFERING_AMT	            250000.0	250000.0
19	  AMOUNT_OUTSTANDING	            250000.0	250000.0
20	         Vol_grt_out	                 0.0	0.0

"""
