from copy import deepcopy
import numpy as np
import torch

from Configurations import *
import Utils

alpha = 0.1

# differential feilds
def mdsg_apf(node_global_positions, remain_list, dimension, d, khop=3):
    """
    fly to the center of the swarm is thinks
    :param node_global_positions:
    :return:
    """
    speed = []
    remain_positions = []
    damage_positions = []

    damage_list = [i for i in range(len(node_global_positions)) if i not in remain_list]
    
    for i in remain_list:
        remain_positions.append(deepcopy(node_global_positions[i]))
    for i in damage_list:
        damage_positions.append(deepcopy(node_global_positions[i]))

    remain_positions = np.array(remain_positions)
    damage_positions = np.array(damage_positions)

    # final_positions = np.mean(remain_positions, 0)
    # final_positions = np.array([500, 500]) if dimension == 2 else np.array([500, 500, 50])
    final_positions = np.mean(damage_positions, axis=0)

    all_neighbour = Utils.calculate_khop_neighbour(node_global_positions, d)

    for i in range(len(node_global_positions)):
        if i not in remain_list:
            speed.append([0 for _ in range(dimension)])
            continue

        self_positions = node_global_positions[i]
        
        damage_neighbour_directions = []
        for k in range(khop):
            if k <= len(all_neighbour[i]):
                for node in all_neighbour[i][k]:
                    damage_neighbour_directions.append(node_global_positions[node] - self_positions)

        damage_neighbour_directions = np.array(damage_neighbour_directions)
        
        flying_direction = alpha * np.mean(damage_neighbour_directions, axis=0) + (1-alpha) * (final_positions - self_positions)
        # flying_direction = (config_communication_range/2)*np.mean(damage_neighbour_directions, axis=0)/np.linalg.norm(np.mean(damage_neighbour_directions, axis=0)) + (final_positions - self_positions)
        # print(np.linalg.norm(np.mean(damage_neighbour_directions, axis=0)), np.linalg.norm(final_positions - self_positions))
        speed.append(flying_direction)

    speed = np.array(speed)
    max_dis = np.max(np.linalg.norm(speed, axis=1))
    speed = np.array(speed / max_dis)
    # print(len(speed))
    return deepcopy(speed)

def mdsg_apf_khop(node_global_positions, remain_list, dimension, d, khoplist=4):
    """
    fly to the center of the swarm is thinks
    :param node_global_positions:
    :return:
    """
    speed = []
    remain_positions = []
    damage_positions = []

    damage_list = [i for i in range(len(node_global_positions)) if i not in remain_list]
    
    for i in remain_list:
        remain_positions.append(deepcopy(node_global_positions[i]))
    for i in damage_list:
        damage_positions.append(deepcopy(node_global_positions[i]))

    remain_positions = np.array(remain_positions)
    damage_positions = np.array(damage_positions)

    # final_positions = np.mean(remain_positions, 0)
    # final_positions = np.array([500, 500]) if dimension == 2 else np.array([500, 500, 50])
    final_positions = np.mean(damage_positions, axis=0)

    all_neighbour = Utils.calculate_khop_neighbour(node_global_positions, d)

    FloatTensor = torch.cuda.FloatTensor

    speed_list, step_list = [], []

    for khop in range(1, khoplist+1):
        speed = []

        for i in range(len(node_global_positions)):
            if i not in remain_list:
                speed.append([0 for _ in range(dimension)])
                continue

            self_positions = node_global_positions[i]
            
            damage_neighbour_directions = []
            for k in range(khop):
                if k <= len(all_neighbour[i]):
                    for node in all_neighbour[i][k]:
                        damage_neighbour_directions.append(node_global_positions[node] - self_positions)

            damage_neighbour_directions = np.array(damage_neighbour_directions)
            
            flying_direction = alpha * np.mean(damage_neighbour_directions, axis=0) + (1-alpha) * (final_positions - self_positions)
            # flying_direction = (config_communication_range/2)*np.mean(damage_neighbour_directions, axis=0)/np.linalg.norm(np.mean(damage_neighbour_directions, axis=0)) + (final_positions - self_positions)
            # print(np.linalg.norm(np.mean(damage_neighbour_directions, axis=0)), np.linalg.norm(final_positions - self_positions))
            speed.append(flying_direction)

        speed = np.array(speed)
        max_dis = np.max(np.linalg.norm(speed, axis=1))
        speed = np.array(speed / max_dis)

        speed_list.append(deepcopy(speed))

        speed = np.array([speed[i] for i in remain_list])

        speed = torch.FloatTensor(speed).type(FloatTensor)
        remain_positions_torch = torch.FloatTensor(deepcopy(remain_positions)).type(FloatTensor)

        for step in range(500):
            try:
                e_vals, num = Utils.check_number_of_clusters_torch(remain_positions_torch, config_communication_range)
            except:
                final_positions_ = remain_positions_torch.cpu().data.numpy()
                A = Utils.make_A_matrix(final_positions_, len(final_positions_), config_communication_range)
                D = Utils.make_D_matrix(A, len(A))
                L = D - A
                flag, num = Utils.check_number_of_clusters(L, len(L))
            # print(khop, step, num)
            if num == 1:
                break

            remain_positions_torch += speed

        step_list.append(step)

    # with open(f'./Logs/khop-old/MDSG-APF_d{200-len(remain_list)}.txt', 'a') as f:
    #     print(step_list, file=f)

    best_speed = speed_list[np.argmin(step_list)]
    return deepcopy(best_speed)