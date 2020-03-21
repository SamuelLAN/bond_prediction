import os
import time

# MODEL_NAME = 'transformer'
MODEL_NAME = 'transformer_no_embeddings'

TIME_DIR = time.strftime('%Y_%m_%d_%H_%M_%S')
# TIME_DIR = '2020_03_15_18_19_02'
IS_TRAIN = True

__group_dir = 'D:\Data\share_mine_laptop\community_detection\data\inputs'
group_name = 'group_Spectral_Clustering_filter_lower_5_with_model_input_features'
# group_name = 'group_K-means_filter_lower_5'
# group_name = 'group_Spectral_Clustering_filter_lower_5'
group_param_name = 'no_day_off_no_distinguish_buy_sell_use_transaction_count'
group_file_name = 'group_0_no_below_50_25_10_g_minus_1_1'
group_path = os.path.join(__group_dir, group_name, group_param_name, group_file_name)

LOG = {
    'group_name': group_name,
    'group_param_name': group_param_name,
    'group_file_name': group_file_name,
}
