from copy import deepcopy
import Utils
from Configurations import *
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

best_hidden_dimension = 512
best_dropout = 0.1
# lr = 0.00001
alpha = 0.12
draw = False

dimension = config_dimension

torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)

class DEMD:
    def __init__(self):
        self.hidden_dimension = best_hidden_dimension
        self.dropout_value = best_dropout

    def demd(self, global_positions, remain_list, khop=4):
        gcn_network = GCN_diffussion_structure(nfeat=dimension, nhid=self.hidden_dimension, nclass=dimension, dropout=self.dropout_value, if_dropout=True, bias=True)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            gcn_network.cuda()

        # self.optimizer = Adam(self.gcn_network.parameters(), lr=0.001)
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        optimizer = torch.optim.Adam(gcn_network.parameters(), lr=0.0001)

        remain_positions = []
        for i in remain_list:
            remain_positions.append(deepcopy(global_positions[i]))
        fixed_positions = deepcopy(remain_positions)
        remain_positions = np.array(remain_positions)
        num_remain = len(remain_list)

        # my
        A = make_khop_A_matrix(global_positions, remain_list, config_communication_range, khop)
        D = Utils.make_D_matrix(A, num_remain)
        L = D - A
        A_norm = np.linalg.norm(A, ord=np.inf)
        k0 = 1 / A_norm
        K = 0.99 * k0
        A_hat = np.eye(num_remain) - K * L

        central_point = np.array([500, 500]) if dimension == 2 else np.array([500, 500, 50])
        
        central_directions = central_point - remain_positions
        central_directions = central_directions / np.linalg.norm(central_directions, axis=1).reshape(num_remain, -1)
        central_directions = torch.FloatTensor(central_directions).type(FloatTensor)

        central_distence = np.linalg.norm(central_point - remain_positions, axis=1).reshape(num_remain, -1)
        central_distence = torch.FloatTensor(central_distence).type(FloatTensor)
        # print(central_directions)

        A_init = deepcopy(A)

        for i in range(len(global_positions)):
            if i not in remain_list:
                fixed_positions.append(deepcopy(global_positions[i]))
        # fixed_positions.append(np.array([500, 500]))

        fixed_positions = np.array(fixed_positions)
        A_fixed = Utils.make_A_matrix(fixed_positions, len(fixed_positions), config_communication_range)
        fixed_positions = torch.FloatTensor(fixed_positions).type(FloatTensor)
        A_fixed = torch.FloatTensor(A_fixed).type(FloatTensor)
        # my end

        remain_positions = torch.FloatTensor(remain_positions).type(FloatTensor)
        A_hat = torch.FloatTensor(A_hat).type(FloatTensor)
        best_final_positions = 0
        best_loss = 1000000000000
        loss_ = 0
        counter_loss = 0
        
        # print("---------------------------------------")
        # print("start training GCN ... ")
        # print("=======================================")
        for train_step in range(1000):
            if loss_ > 1000 and train_step > 50:
                optimizer = torch.optim.Adam(gcn_network.parameters(), lr=0.0001)
            if counter_loss > 4 and train_step > 50:
                break
            
            # final_positions = self.gcn_network(remain_positions, A_hat)
            final_positions = gcn_network(fixed_positions, A_hat, A_fixed, num_remain)

            # if dimension == 3:
            #     final_positions = 0.5 * torch.Tensor(np.array([config_width, config_length, config_height])).type(self.FloatTensor) * final_positions
            # else:
            #     final_positions = 0.5 * torch.Tensor(np.array([config_width, config_length])).type(self.FloatTensor) * final_positions
            # if train_step == 0: print(final_positions)

            # check if connected
            final_positions_ = final_positions.cpu().data.numpy()
            # A = Utils.make_A_matrix(final_positions_, len(final_positions_), config_communication_range)
            # D = Utils.make_D_matrix(A, len(A))
            # L = D - A
            # flag, num_ = Utils.check_number_of_clusters(L, len(L))
            _, num = Utils.check_number_of_clusters_torch(final_positions, config_communication_range)
            # print(num_, num)

            # loss
            loss =[]

            # loss1
            loss.append(torch.max(torch.norm(final_positions-remain_positions, dim=1)))
            # loss.append(torch.sum(torch.norm(final_positions-remain_positions, dim=1))*10/num_remain)
            # loss.append(torch.norm(torch.norm(final_positions-remain_positions, dim=1)))
            # print(final_positions)

            # loss2
            degree = torch.Tensor(np.sum(A_init, axis=0) / np.sum(A_init)).type(FloatTensor).reshape((num_remain,1))
            centroid = torch.sum(torch.mul(final_positions,degree), dim=0)
            centrepoint = 0.5*torch.max(remain_positions, dim=0)[0] + 0.5*torch.min(remain_positions, dim=0)[0]
            # centrepoint = torch.Tensor([500,500]).type(FloatTensor)
            # print(centroid, centrepoint)
            loss.append(torch.norm(centroid - centrepoint))

            # loss 3
            directions = final_positions - remain_positions
            directions = directions / torch.reshape(torch.norm(directions, dim=1), (num_remain, 1))
            directions_loss = (directions - central_directions) * central_distence / torch.Tensor(np.sum(A_init, axis=0)).type(FloatTensor).reshape((num_remain,1))
            # if train_step < 50:
            #     loss.append(torch.norm(directions_loss))

            loss = torch.stack(loss)
            # print(loss)

            # initialization
            if train_step == 0:
                loss_weights = torch.ones_like(loss)
                # loss_weights = torch.tensor((1,1,10)).cuda()
                loss_weights = torch.nn.Parameter(loss_weights)
                T = loss_weights.sum().detach()
                loss_optimizer = torch.optim.Adam([loss_weights], lr=0.01)
                l0 = loss.detach()
                layer = gcn_network.gc8

            # compute the weighted loss
            weighted_loss = 1000*(num-1) + loss @ loss_weights
            # weighted_loss = 1000 * (num - 1) + loss_weights @ loss + torch.var(degree-degree_init)
            # print(weighted_loss.cpu().data.numpy())
            if weighted_loss.cpu().data.numpy() < best_loss:
                best_loss = deepcopy(weighted_loss.cpu().data.numpy())
                best_final_positions = deepcopy(final_positions.cpu().data.numpy())

            # clear gradients of network
            optimizer.zero_grad()
            # backward pass for weigthted task loss
            weighted_loss.backward(retain_graph=True)
            # compute the L2 norm of the gradients for each task
            gw = []
            for i in range(len(loss)):
                dl = torch.autograd.grad(loss_weights[i]*loss[i], layer.parameters(), retain_graph=True, create_graph=True)[0]
                gw.append(torch.norm(dl))
            gw = torch.stack(gw)
            # compute loss ratio per task
            loss_ratio = loss.detach() / l0
            # compute the relative inverse training rate per task
            rt = loss_ratio / loss_ratio.mean()
            # compute the average gradient norm
            gw_avg = gw.mean().detach()
            # compute the GradNorm loss
            constant = (gw_avg * rt ** alpha).detach()
            gradnorm_loss = torch.abs(gw - constant).sum()
            # clear gradients of loss_weights
            loss_optimizer.zero_grad()
            # backward pass for GradNorm
            gradnorm_loss.backward()

            # log loss_weights and loss
            # update model loss_weights
            optimizer.step()
            # update loss loss_weights
            loss_optimizer.step()

            # renormalize loss_weights
            loss_weights = (loss_weights / loss_weights.sum() * T).detach()
            loss_weights = torch.nn.Parameter(loss_weights)
            loss_optimizer = torch.optim.Adam([loss_weights], lr=0.01)
            
            loss_ = weighted_loss.cpu().data.numpy()
            print(f"episode {train_step}, num {num.cpu().data.numpy()}, loss {loss_}, weights {loss_weights.cpu().data.numpy()}", end='\r')
            # print(torch.norm(final_positions[max_index] - remain_positions[max_index]), torch.norm(centroid - centrepoint))
            # print("---------------------------------------")
            
            if best_loss < 600 and num > 1 and train_step > 50:
                counter_loss += 1
            else:
                counter_loss = 0
            

        speed = np.zeros((config_num_of_agents, dimension))
        remain_positions_numpy = remain_positions.cpu().data.numpy()
        temp_max_distance = 0
        # print("=======================================")

        temp_max_distance = np.max(np.linalg.norm(best_final_positions - remain_positions_numpy, axis=1))
        # print("max_distance", temp_max_distance)
        for i in range(num_remain):
            if np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i]) > 0:
                # speed[remain_list[i]] = (best_final_positions[i] - remain_positions_numpy[i]) / temp_max_distance
                speed[remain_list[i]] = (best_final_positions[i] - remain_positions_numpy[i]) / np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i])


        max_time = temp_max_distance / config_constant_speed

        if draw:
            remain_positions_ = remain_positions.cpu().data.numpy()

            plt.scatter(remain_positions_[:,0], remain_positions_[:,1], c='black')
            plt.scatter(best_final_positions[:,0], best_final_positions[:,1], c='g')
            plt.text(10, 10, f'best time: {max_time}')
            plt.xlim(0, 1000)
            plt.ylim(0, 1000)
            plt.show()
        # print(max_time)

        return deepcopy(speed), deepcopy(max_time), deepcopy(best_final_positions)
    
    def demd_adaptive(self, global_positions, remain_list):
        best_speed, best_max_time, best_positions = [], 100000, []

        for k in range(4, 9):
            speed, max_time, positions = self.demd(global_positions, remain_list, k)
            print(f'khop = {k} with max step {max_time}')
            if max_time < best_max_time:
                best_speed, best_max_time, best_positions = deepcopy(speed), deepcopy(max_time), deepcopy(positions)
            # else:
            #     break

        print(best_max_time)
        return deepcopy(best_speed), deepcopy(best_max_time), deepcopy(best_positions)


def make_khop_A_matrix(global_positions, remain_list, d, khop):
    all_neighbor = calculate_global_neighbour(global_positions, d)

    num_of_agents = len(remain_list)
    A = np.zeros((num_of_agents, num_of_agents))

    for i in range(num_of_agents):
        for j in range(i, num_of_agents):
            if j == i:
                A[i, j] = 0
            else:
                hop = 1
                while(remain_list[j] not in all_neighbor[remain_list[i]][hop-1]):hop+=1
                if hop <= khop:
                    A[i, j] = 1
                    A[j, i] = 1

    return deepcopy(A)

def make_adapted_khop_A_matrix(global_positions, remain_list, d):
    all_neighbor = calculate_global_neighbour(global_positions, d)

    num_of_agents = len(remain_list)

    for k in range(2,19):   
        A = np.zeros((num_of_agents, num_of_agents))
        
        for i in range(num_of_agents):
            for j in range(i, num_of_agents):
                if j == i:
                    A[i, j] = 0
                else:
                    hop = 1
                    while(remain_list[j] not in all_neighbor[remain_list[i]][hop-1]):hop+=1
                    if hop <= k:
                        A[i, j] = 1
                        A[j, i] = 1
        
        D = Utils.make_D_matrix(A, len(A))
        L = D - A
        flag, num = Utils.check_number_of_clusters(L, len(L))

        if num == 1:
            print(k)
            khop = k+3
            break

    # print(k)

    return make_khop_A_matrix(global_positions, remain_list, d, khop)


def calculate_global_neighbour(global_positions, d):
    num_of_agents = len(global_positions)
    neighbour_i_1hop = []
    for i in range(num_of_agents):
        neighbour = []
        for j in range(num_of_agents):
            if i==j: continue

            if np.linalg.norm(global_positions[i, :] - global_positions[j, :]) <= d:
                neighbour.append(j)

        neighbour_i_1hop.append(deepcopy(neighbour))

    all_neighbour = []
    for i in range(num_of_agents):
        cnt = 0
        neighbour_i_all = []
        neighbour_i_multihop = deepcopy(neighbour_i_1hop[i])

        # while len(sum(neighbour_i_all, [])) < 199:
        while len(neighbour_i_multihop) > 0:
            neighbour_i_all.append(deepcopy(neighbour_i_multihop))
            neighbour_i_multihop = []

            for j in neighbour_i_all[cnt]:
                for k in neighbour_i_1hop[j]:
                    if k not in sum(neighbour_i_all, []) and k not in neighbour_i_multihop and k != i:
                        neighbour_i_multihop.append(k)

            # neighbour_i_all.append(deepcopy(neighbour_i_multihop))
            cnt += 1
            
        # print(neighbour_i_all)
        all_neighbour.append(deepcopy(neighbour_i_all))

    return deepcopy(all_neighbour)


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_diffussion_structure(nn.Module):
    def __init__(self, nfeat=3, nhid=5, nclass=3, dropout=0.5, if_dropout=True, bias=True):
        super(GCN_diffussion_structure, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, bias=bias)
        self.gc2 = GraphConvolution(nhid, nhid, bias=bias)
        self.gc3 = GraphConvolution(nhid, nhid, bias=bias)
        self.gc4 = GraphConvolution(nhid, nhid, bias=bias)
        self.gc5 = GraphConvolution(nhid, nhid, bias=bias)
        self.gc6 = GraphConvolution(nhid, nhid, bias=bias)
        self.gc7 = GraphConvolution(nhid, nhid, bias=bias)
        self.gc8 = GraphConvolution(nhid, nclass, bias=bias)
        self.dropout = dropout
        self.training = if_dropout

    def forward(self, x, adj, adj_d, num_remain):
        x = F.relu(self.gc1(x, adj_d))
        x = x.split(num_remain, dim=0)[0]
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc6(x, adj))
        x = F.relu(self.gc7(x, adj))
        x = self.gc8(x, adj)

        # return torch.tanh(x) + 1
        return x