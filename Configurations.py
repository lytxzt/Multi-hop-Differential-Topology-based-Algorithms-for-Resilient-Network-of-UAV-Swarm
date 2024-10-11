import pandas as pd
import numpy as np
import random
import torch

"""
specify a certain GPU
"""
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'

# random seed
np.random.seed(1)
random.seed(2)
torch.manual_seed(1)

config_dimension = 2
config_initial_swarm_positions = pd.read_excel("Configurations/swarm_positions_200.xlsx")
config_initial_swarm_positions = config_initial_swarm_positions.values[:, 1:1+config_dimension]
config_initial_swarm_positions = np.array(config_initial_swarm_positions, dtype=np.float64)

# configurations on swarm
config_num_of_agents = 200
config_communication_range = 120

# configurations on environment
config_width = 1000.0
config_length = 1000.0
config_height = 100.0

config_constant_speed = 1

# configurations on destroy
config_maximum_destroy_num = 50
config_minimum_remain_num = 5

# configurations on meta learning
config_meta_training_epi = 500
# configurations on Graph Convolutional Network
config_K = 1 / 100
config_best_eta = 0.3
config_best_epsilon = 0.99

# configurations on one-off UEDs
config_num_destructed_UAVs = 50  # should be in the range of [1, num_of_UAVs-2]
config_normalize_positions = True

# configurations on training GCN
config_alpha_k = [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 0.9, 0.95, 1, 1.5, 2, 3, 5]
config_gcn_repeat = 100
config_expension_alpha = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
config_d0_alpha = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

config_representation_step = 450

config_random_seed = [927, 1501, 1391, 5916, 2771, 5049, 4122, 9928, 3477, 9942, 586, 9523, 2595, 7057, 6448, 8341, 6096, 8916, 7289, 8226, 4395, 589, 450, 5965, 7617, 5218, 6227, 6941, 8614, 2695, 9184, 2908, 3869, 3779, 391, 2896, 5328, 2845, 2240, 8359, 8360, 5894, 8418, 9174, 2980, 7302, 6794, 8608, 5968, 9724, 5797, 5930, 7304, 2641, 6552, 7560, 8690, 4095, 8029, 4573, 8161, 8206, 8445, 5799, 7450, 7554, 5748, 9302, 9136, 7481, 7973, 3635, 5320, 2721, 4394, 7861, 5072, 4970, 8262, 9211, 8483, 8313, 9633, 6663, 5110, 3405, 8011, 8387, 6007, 1235, 5595, 138, 3136, 1740, 963, 9412, 802, 4475, 9695, 3713, 1742, 8559, 2237, 4356, 4012, 3449, 990, 6930, 523, 931, 5937, 5902, 2817, 4088, 385, 1359, 1888, 1106, 416, 670, 347, 6113, 4190, 2094, 2575, 3011, 8571, 32, 6318, 9658, 708, 4061, 2481, 595, 69, 5640, 1854, 4687, 5525, 8008, 505, 5053, 7351, 9036, 9915, 750, 4325, 6584, 2515, 7747, 3696, 1532, 5183, 1672, 397, 7338, 2090, 8492, 9584, 6439, 7978, 8435, 5374, 2357, 5588, 4246, 4290, 9930, 6878, 296, 9142, 2304, 4145, 550, 2158, 2797, 1571, 7429, 3796, 8328, 515, 4043, 3809, 7286, 1206, 4109, 1318, 9687, 3739, 5896, 4205, 6931, 4567, 8622, 80, 2476, 582, 6304, 6697, 2626, 1822, 8390, 1439, 3947, 1670, 1635, 325, 2978, 3794, 1724, 3562, 401, 8532, 7609, 7437, 5075, 8775, 6226, 3481, 3443, 7107, 6974, 8381, 350, 9521, 9689, 840, 6849, 8603, 9525, 2970, 1537, 7862, 6000, 320, 8507, 1943, 6005, 4745, 6099, 5051, 313, 6755, 1658, 1721, 5013, 3251, 258, 7397, 983]

