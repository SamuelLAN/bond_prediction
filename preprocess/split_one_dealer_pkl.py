import os
from lib import utils
import random

dir_path = r'D:\Data\share_mine_laptop\community_detection\data\input_data_dealer_prediction\group_k_means_cluster_4_feat_1_trace_count_2_volume_3_num_dealer_split_by_date\no_day_off_no_distinguish_buy_sell_use_transaction_count\group_3'
new_dir_path = r'D:\Data\share_mine_laptop\community_detection\data\input_data_dealer_prediction\group_k_means_cluster_4_feat_1_trace_count_2_volume_3_num_dealer_split_by_date\no_day_off_no_distinguish_buy_sell_use_transaction_count\group_3_v1'
batch_size = 12

if not os.path.exists(new_dir_path):
    os.mkdir(new_dir_path)

file_list = os.listdir(dir_path)
length = len(file_list)
for i, file_name in enumerate(file_list):
    if i % 2 == 0:
        progress = float(i + 1) / length * 100.
        print('\rprogress: %.2f%% ' % progress, end='')

    file_path = os.path.join(dir_path, file_name)
    file_name_prefix = os.path.splitext(file_name)[0]

    x, y = utils.load_pkl(file_path)
    data = list(zip(x, y))

    random.seed(42)
    random.shuffle(data)

    file_no = 0
    while len(data):
        new_file_path = os.path.join(new_dir_path, f'{file_name_prefix}_f{file_no}.pkl')
        file_no += 1

        tmp_data = data[:batch_size]
        tmp_x, tmp_y = list(zip(*tmp_data))
        utils.write_pkl(new_file_path, [tmp_x, tmp_y])

        data = data[batch_size:]

    os.remove(file_path)
