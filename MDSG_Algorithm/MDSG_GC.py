
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
from copy import deepcopy

import Utils
from MDSG_Algorithm.Bipartite_GCN import Bipartite_GCN_structure
from Configurations import *

best_hidden_dimension = 512
best_dropout = 0.1
# lr = 0.00001
alpha = 0.12
draw = False

dimension = config_dimension

class MDSG_GC:
    def __init__(self):
        self.hidden_dimension = best_hidden_dimension
        self.dropout_value = best_dropout

    def mdsg_gc(self, global_positions, remain_list, khop=3):
        gcn_network = Bipartite_GCN_structure(nfeat=dimension, nhid=self.hidden_dimension, nclass=dimension, dropout=self.dropout_value, if_dropout=True, bias=True)

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            gcn_network.cuda()

        # self.optimizer = Adam(self.gcn_network.parameters(), lr=0.001)
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        optimizer = Adam(gcn_network.parameters(), lr=0.001)

        remain_positions = []
        for i in remain_list:
            remain_positions.append(deepcopy(global_positions[i]))
        
        num_remain = len(remain_list)
        
        fixed_positions = deepcopy(remain_positions)
        for i in range(len(global_positions)):
            if i not in remain_list:
                fixed_positions.append(deepcopy(global_positions[i]))
        fixed_positions.append(np.array([500, 500]))

        num_of_agents = len(fixed_positions)
        
        fixed_positions = np.array(fixed_positions)
        remain_positions = np.array(remain_positions)

        # damage differential
        all_neighbor = Utils.calculate_khop_neighbour(fixed_positions, config_communication_range)
        A = Utils.make_khop_A_matrix(all_neighbor, fixed_positions, num_remain, khop)
        # A = Utils.make_A_matrix(remain_positions, num_remain, config_communication_range)

        D = Utils.make_D_matrix(A, num_of_agents)
        # L = D - A
        # A_norm = np.linalg.norm(A, ord=np.inf)
        # k0 = 1 / A_norm
        # K = 0.99 * k0
        # A_hat = np.eye(num_of_agents) - K * L
        
        D_tilde_sqrt = np.diag(D.diagonal() ** (-0.5))
        A_hat = np.eye(num_of_agents) - D_tilde_sqrt @ A @ D_tilde_sqrt
        
        A_init = deepcopy(A)

        fixed_positions = torch.FloatTensor(fixed_positions).type(FloatTensor)
        remain_positions = torch.FloatTensor(remain_positions).type(FloatTensor)
        A_hat = torch.FloatTensor(A_hat).type(FloatTensor)

        best_final_positions = 0
        best_loss = 1000000000000
        loss_ = 0

        # print("---------------------------------------")
        # print("start training GCN ... ")
        # print("=======================================")
        for train_step in range(1000):
            # print(train_step)

            final_positions = gcn_network(fixed_positions, A_hat, num_remain)


            if dimension == 3:
                final_positions = 0.5 * torch.Tensor(np.array([config_width, config_length, config_height])).type(FloatTensor) * final_positions
            else:
                final_positions = 0.5 * torch.Tensor(np.array([config_width, config_length])).type(FloatTensor) * final_positions
            # if train_step == 0: print(final_positions)

            # check if connected
            final_positions_ = final_positions.cpu().data.numpy()
            # A = Utils.make_A_matrix(final_positions_, len(final_positions_), config_communication_range)
            # D = Utils.make_D_matrix(A, len(A))
            # L = D - A
            # flag, num = Utils.check_number_of_clusters(L, len(L))
            e_vals, num = Utils.check_number_of_clusters_torch(final_positions, config_communication_range)
            # print(num)

            # loss
            loss =[]

            # loss1
            loss.append(torch.max(torch.norm(final_positions-remain_positions, dim=1)))
            # loss.append(torch.sum(torch.norm(final_positions-remain_positions, dim=1))*10/num_remain)
            # loss.append(torch.norm(torch.norm(final_positions-remain_positions, dim=1)))
            # print(final_positions)
            # loss.append(torch.norm(e_vals))

            # loss2
            degree = torch.Tensor(np.sum(A_init, axis=0) / np.sum(A_init)).type(FloatTensor).reshape((num_of_agents,1))
            degree = degree.split(num_remain, dim=0)[0]
            centroid = torch.sum(torch.mul(final_positions, degree), dim=0)
            # centroid = torch.mean(final_positions, dim=0)

            # centrepoint = 0.5*torch.max(final_positions, dim=0)[0] + 0.5*torch.min(final_positions, dim=0)[0]
            centrepoint = 0.5*torch.max(remain_positions, dim=0)[0] + 0.5*torch.min(remain_positions, dim=0)[0]

            # print(centroid, centrepoint)
            # loss.append(torch.norm(centroid - centrepoint))
            
            # print(loss)

            loss = torch.stack(loss)
            # print(loss)

            # initialization
            if train_step == 0:
                loss_weights = torch.ones_like(loss)
                # loss_weights = torch.tensor((1,1,10)).cuda()
                loss_weights = torch.nn.Parameter(loss_weights)
                T = loss_weights.sum().detach()
                loss_optimizer = Adam([loss_weights], lr=0.01)
                l0 = loss.detach()
                # layer = gcn_network.out_att
                layer = gcn_network.gc8


            # compute the weighted loss
            weighted_loss = 50000*(num - 1) + loss_weights @ loss
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

            if draw and train_step % 200 == 0:
                remain_positions_ = remain_positions.cpu().data.numpy()
                final_positions_ = final_positions.cpu().data.numpy()

                plt.scatter(remain_positions_[:,0], remain_positions_[:,1], c='black')
                plt.scatter(final_positions_[:,0], final_positions_[:,1], c='g')
                plt.text(10, 10, f'best loss: {loss_}')
                # plt.xlim(0, 1000)
                # plt.ylim(0, 1000)
                plt.show()

        speed = np.zeros((config_num_of_agents, dimension))
        remain_positions_numpy = remain_positions.cpu().data.numpy()
        temp_max_distance = 0
        # print("=======================================")

        for i in range(num_remain):
            if np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i]) > 0:
                speed[remain_list[i]] = (best_final_positions[i] - remain_positions_numpy[i]) / np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i])
            if np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i]) > temp_max_distance:
                temp_max_distance = deepcopy(np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i]))

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

    def mdsg_gc_adaptive(self, global_positions, remain_list):
        best_speed, best_max_time, best_positions = [], 100000, []
        best_k = 1

        for k in range(4, 7):
            speed, max_time, positions = self.mdsg_gc(global_positions, remain_list, k)
            A = Utils.make_A_matrix(positions, len(positions), config_communication_range)
            D = Utils.make_D_matrix(A, len(A))
            L = D - A
            flag, num = Utils.check_number_of_clusters(L, len(L))
            print(f'khop = {k} with num {num}, max step {max_time}')
            if max_time < best_max_time and num == 1:
                best_speed, best_max_time, best_positions = deepcopy(speed), deepcopy(max_time), deepcopy(positions)
                best_k = k
            # else:
            #     break

        print(best_k, best_max_time)
        return deepcopy(best_speed), deepcopy(best_max_time), deepcopy(best_positions)