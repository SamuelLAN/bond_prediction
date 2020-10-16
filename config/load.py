import os
import time

MODEL_NAME = 'transformer2'
# MODEL_NAME = 'transformer_no_embeddings'

# TIME_DIR = time.strftime('%Y_%m_%d_%H_%M_%S')
TIME_DIR = '2020_04_05_12_57_23'
IS_TRAIN = False

__group_dir = r'D:\Data\share_mine_laptop\community_detection\data\input_data'
# group_name = r'group_Spectral_Clustering_filter_lower_5_with_model_input_features'
group_name = 'group_k_means_split_by_date'
# group_name = 'group_spectral_clustering_with_patterns_info_cluster_6_split_by_date'
group_param_name = 'no_day_off_no_distinguish_buy_sell_use_transaction_count'
group_file_name = 'group_3'
group_path = os.path.join(__group_dir, group_name, group_param_name, group_file_name)

LOG = {
    'group_name': group_name,
    'group_param_name': group_param_name,
    'group_file_name': group_file_name,
}
