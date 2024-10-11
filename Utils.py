import numpy as np
import pandas as pd
import torch
from copy import deepcopy
from operator import itemgetter
from Configurations import *


def make_A_matrix(positions, num_of_agents, d):
    A = np.zeros((num_of_agents, num_of_agents))
    for i in range(num_of_agents):
        for j in range(i, num_of_agents):
            if j == i:
                A[i, j] = 0
            else:
                distance = np.linalg.norm(positions[i, :] - positions[j, :])
                if distance <= d:
                    A[i, j] = 1
                    A[j, i] = 1
    return deepcopy(A)


def make_D_matrix(A, num_of_agents):
    D = np.zeros((num_of_agents, num_of_agents))
    for i in range(num_of_agents):
        D[i, i] = np.sum(A[i])
    return deepcopy(D)


def check_number_of_clusters(L, num_of_agents):
    e_vals, e_vecs = np.linalg.eig(L)
    eig_0_counter = 0
    for i in range(num_of_agents):
        if e_vals[i] < 0.000001:
            eig_0_counter += 1
    return eig_0_counter == 1, eig_0_counter


def check_number_of_clusters_torch(positions, d):
    m, n = positions.shape
    G = torch.mm(positions, positions.T)
    H = torch.tile(torch.diag(G), (m, 1))
    D = torch.sqrt(H + H.T - 2*G)
    # print(m, n, D)
    A = torch.where(D > d, 0, 1.0)
    # print(m, n, A)
    D = torch.diag(torch.sum(A, dim=1))
    # print(m, n, D)
    L = D - A

    e_vals, e_vecs = torch.linalg.eigh(L)
    # print(e_vals.real)
        
    num = torch.sum(torch.where(e_vals.real < 0.0001, 1, 0))
    # print(torch.sum(e_vals), torch.trace(L), torch.sum(A))
    return e_vals.real, num


def calculate_d_max(positions):
    d_max = 0
    for i in range(len(positions) - 1):
        for j in range(i + 1, len(positions)):
            if d_max < np.linalg.norm(positions[j] - positions[i]):
                d_max = deepcopy(np.linalg.norm(positions[j] - positions[i]))

    return deepcopy(d_max)


def check_if_a_connected_graph(remain_positions, remain_num):
    A = make_A_matrix(remain_positions, remain_num, config_communication_range)
    D = make_D_matrix(A, remain_num)
    L = D - A
    connected_flag, num_of_clusters = check_number_of_clusters(L, remain_num)
    return deepcopy(connected_flag), deepcopy(num_of_clusters)


def split_the_positions_into_clusters(positions, num_of_clusters, A):
    positions_with_clusters = []
    remain_list = [i for i in range(len(positions))]
    if num_of_clusters <= 1:
        return None
    else:
        for k in range(num_of_clusters):
            temp_positions = []

            visited = np.zeros(len(remain_list))
            counter = 0
            stack = Stack()
            stack.push(remain_list[0])
            visited[0] = 1
            counter += 1

            while stack.length() != 0:
                current = stack.top_element()
                flag = True
                temp_counter = 0
                for i in remain_list:
                    if A[current, i] == 1 and visited[temp_counter] == 0:
                        visited[temp_counter] = 1
                        counter += 1
                        stack.push(i)
                        flag = False
                        break
                    temp_counter += 1
                if flag:
                    stack.pop()

            visited_node = []
            for j in range(len(remain_list)):
                if visited[j] == 1:
                    visited_node.append(remain_list[j])
            for j in range(counter):
                remain_list.remove(visited_node[j])
                temp_positions.append(deepcopy(positions[visited_node[j]]))
            positions_with_clusters.append(deepcopy(np.array(temp_positions)))
        return deepcopy(positions_with_clusters)


class Stack:
    def __init__(self):
        self.memory = []

    def push(self, num):
        self.memory.append(num)

    def pop(self):
        if len(self.memory) == 0:
            return None
        else:
            temp = self.memory[-1]
            del self.memory[-1]
            return temp

    def length(self):
        return len(self.memory)

    def top_element(self):
        return self.memory[-1]


def split_the_positions_into_clusters_and_indexes(positions, num_of_clusters, A):
    positions_with_clusters = []
    cluster_index = []
    remain_list = [i for i in range(len(positions))]
    if num_of_clusters <= 1:
        positions_with_clusters.append(deepcopy(positions))
        cluster_index.append(remain_list)
        return deepcopy(positions_with_clusters), deepcopy(cluster_index)
    else:
        for k in range(num_of_clusters):
            temp_positions = []
            temp_index = []

            visited = np.zeros(len(remain_list))
            counter = 0
            stack = Stack()
            stack.push(remain_list[0])
            visited[0] = 1
            counter += 1

            while stack.length() != 0:
                current = stack.top_element()
                flag = True
                temp_counter = 0
                for i in remain_list:
                    if A[current, i] == 1 and visited[temp_counter] == 0:
                        visited[temp_counter] = 1
                        counter += 1
                        stack.push(i)
                        flag = False
                        break
                    temp_counter += 1
                if flag:
                    stack.pop()

            visited_node = []
            for j in range(len(remain_list)):
                if visited[j] == 1:
                    visited_node.append(remain_list[j])
                    temp_index.append(remain_list[j])
            for j in range(counter):
                remain_list.remove(visited_node[j])
                temp_positions.append(deepcopy(positions[visited_node[j]]))

            positions_with_clusters.append(deepcopy(np.array(temp_positions)))
            cluster_index.append(deepcopy(temp_index))
        return deepcopy(positions_with_clusters), deepcopy(cluster_index)


def intersection_set(listA, listB):
    intersection_list = [i for i in listA if i in listB]
    return deepcopy(intersection_list)


def difference_set(listA, listB):
    difference_list = [i for i in listA if i not in listB]
    return deepcopy(difference_list)


def union_set(listA, listB):
    union_set = [i for i in listA]
    for i in listB:
        if i not in listA:
            union_set.append(deepcopy(i))
    return deepcopy(union_set)


def smallest_d_algorithm(positions, num, d0):
    A = make_A_matrix(positions, num, d0)
    d_min = deepcopy(d0)
    unsorted_list = []
    dis = []
    sorted_list = []
    for i in range(num - 1):
        for j in range(i + 1, num):
            unsorted_list.append(deepcopy({"start": i,
                                           "end": j,
                                           "distance": np.linalg.norm(positions[i] - positions[j])}))
            dis.append(deepcopy(np.linalg.norm(positions[i] - positions[j])))
    sorted_index = [index for index, value in sorted(enumerate(dis), key=itemgetter(1))]
    for i in range(len(sorted_index)):
        sorted_list.append(unsorted_list[sorted_index[i]])

    # find the threshold
    threshold_for_d0 = 0
    for i in range(len(sorted_index)):
        if sorted_list[i]["distance"] > d_min:
            threshold_for_d0 = deepcopy(i)
            break
    for i in range(threshold_for_d0, len(sorted_index)):
        A[sorted_list[i]["start"], sorted_list[i]["end"]] = 1
        A[sorted_list[i]["end"], sorted_list[i]["start"]] = 1
        D = make_D_matrix(A, num)
        L = D - A
        connected_flag, num_cluster = check_number_of_clusters(L, num)
        if connected_flag:
            d_min = deepcopy(sorted_list[i]["distance"])
            break
    return deepcopy(d_min)

def calculate_khop_neighbour(positions, d):
    num_of_agents = len(positions)
    neighbour_i_1hop = []
    for i in range(num_of_agents):
        neighbour = []
        for j in range(num_of_agents):
            if i==j: continue

            if np.linalg.norm(positions[i, :] - positions[j, :]) <= d:
                neighbour.append(j)

        neighbour_i_1hop.append(deepcopy(neighbour))

    all_neighbour = []
    for i in range(num_of_agents):
        cnt = 0
        neighbour_i_all = []
        neighbour_i_multihop = deepcopy(neighbour_i_1hop[i])

        while len(neighbour_i_multihop) > 0:
            neighbour_i_all.append(deepcopy(neighbour_i_multihop))
            neighbour_i_multihop = []

            for j in neighbour_i_all[cnt]:
                for k in neighbour_i_1hop[j]:
                    if k not in sum(neighbour_i_all, []) and k not in neighbour_i_multihop and k != i:
                        neighbour_i_multihop.append(k)

            cnt += 1
            
        all_neighbour.append(deepcopy(neighbour_i_all))

    return deepcopy(all_neighbour)


def make_khop_A_matrix(all_neighbor, positions, num_remain, khop=5):
    # all_neighbor = calculate_khop_neighbour(positions, d)

    num_of_agents = len(positions)
    A = np.zeros((num_of_agents, num_of_agents))

    for i in range(num_remain):
        for j in range(num_remain, num_of_agents-1):
            hop = 1
            # if np.linalg.norm(positions[j] - central_point) >= embedding_distence : continue
            # dis = np.linalg.norm(positions[j] - central_point) / embedding_distence 
            # dis = np.cos(dis*np.pi/2)
            while(j not in all_neighbor[i][hop-1]):hop+=1
            if hop <= khop:
                A[i, j] = 1
                A[j, i] = 1

    for i in range(num_of_agents-1):
        A[i, num_of_agents-1] = 1
        A[num_of_agents-1, i] = 1

    # print(A)
    return deepcopy(A)