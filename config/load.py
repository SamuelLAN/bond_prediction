import os
import time

# MODEL_NAME = 'transformer_with_feature_vectors'
MODEL_NAME = 'transformer_modified_zero_with_summed_bond_embeddings_only_sell'
# MODEL_NAME = 'transformer_modified_zero_with_summed_bond_embeddings'
# MODEL_NAME = 'transformer_rezero_with_summed_bond_embeddings'
# MODEL_NAME = 'transformer_modified_zero_with_summed_bond_embeddings_kmeans_3'
# MODEL_NAME = 'transformer_modified_zero_vector_init_zero_2_with_summed_bond_embeddings'
# MODEL_NAME = 'transformer_modified_zero_same_v_with_summed_bond_embeddings'
# MODEL_NAME = 'transformer_modified_zero_weight_share_with_summed_bond_embeddings'
# MODEL_NAME = 'transformer_modified_zero_new_loss_with_summed_bond_embeddings'
# MODEL_NAME = 'transformer2'
# MODEL_NAME = 'transformer_no_embeddings'

TIME_DIR = time.strftime('%Y_%m_%d_%H_%M_%S')
# TIME_DIR = '2020_07_19_12_08_33'
# TIME_DIR = '2020_04_17_17_37_12'
IS_TRAIN = True

__group_dir = r'D:\Data\share_mine_laptop\community_detection\data\input_data'
__group_dir_dealer_prediction = r'D:\Data\share_mine_laptop\community_detection\data\input_data_dealer_prediction'
# group_name = r'group_spectral_clustering_with_patterns_info_cluster_6_split_by_date'
# group_name = 'group_k_means_cluster_3_split_by_date'
# group_name = 'group_k_means_cluster_8_split_by_date'
group_name = 'group_k_means_split_by_date'
group_name_dealer_prediction = 'group_k_means_cluster_4_feat_1_trace_count_2_volume_3_num_dealer_split_by_date'
# group_name = 'group_spectral_clustering_with_patterns_info_cluster_6_split_by_date'
# group_param_name = 'no_day_off_no_distinguish_buy_sell_use_transaction_count_only_15_days_input'
group_param_name = 'no_day_off_no_distinguish_buy_sell_use_transaction_count_only_sell'
group_param_name_dealer_prediction = 'no_day_off_no_distinguish_buy_sell_use_transaction_count'
# group_param_name = 'no_day_off_no_distinguish_buy_sell_use_transaction_count_only_buy_y'
# group_param_name = 'no_day_off_no_distinguish_buy_sell_use_transaction_count'
group_file_name = 'group_0'
group_path = os.path.join(__group_dir, group_name, group_param_name, group_file_name)
group_path_dealer_prediction = os.path.join(__group_dir_dealer_prediction, group_name_dealer_prediction, group_param_name_dealer_prediction, group_file_name)

freq_level = 0

LOG = {
    'group_name': group_name,
    'group_param_name': group_param_name,
    'group_file_name': group_file_name,
}
