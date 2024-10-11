import torch
from torch.optim import Adam
from copy import deepcopy

import Utils
from MDSG_Algorithm.Bipartite_GCN import Bipartite_GCN_structure
from Configurations import *

best_hidden_dimension = 512
best_dropout = 0.1
batch_size = 10
# lr = 0.00001
alpha = 0.12

save_loss_curve = False

torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)

'''
mode 1: tahn    + L1
mode 2: no tahn + L1
mode 3: tahn    + no L1
mode 4: no tahn + no L1
'''

dimension = config_dimension
central_point = np.array([500, 500]) if dimension == 2 else np.array([500, 500, 50])

class MDSG_GC_batch:
    def __init__(self, use_pretrained=True):
        self.hidden_dimension = best_hidden_dimension
        self.dropout_value = best_dropout
        self.use_pretrained = use_pretrained

    def mdsg_gc_batch(self, global_positions, remain_list, khop=1):
        gcn_network = Bipartite_GCN_structure(nfeat=dimension, nhid=self.hidden_dimension, nclass=dimension, dropout=self.dropout_value, if_dropout=True, bias=True)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            gcn_network.cuda()

        if self.use_pretrained:
            gcn_network = torch.load("./Pretrained_model/model.pt")
            # gcn_network.load_state_dict(setup_param())

        # self.optimizer = Adam(self.gcn_network.parameters(), lr=0.001)
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        optimizer = Adam(gcn_network.parameters(), lr=0.0001)
        # optimizer = SGD(gcn_network.parameters(), lr=0.01, momentum=0.9)

        remain_positions = []
        for i in remain_list:
            remain_positions.append(deepcopy(global_positions[i]))
        
        num_remain = len(remain_list)
        
        fixed_positions = deepcopy(remain_positions)
        for i in range(len(global_positions)):
            if i not in remain_list:
                fixed_positions.append(deepcopy(global_positions[i]))
        fixed_positions.append(central_point)

        num_of_agents = len(fixed_positions)
        num_destructed = num_of_agents - num_remain
        
        fixed_positions = np.array(fixed_positions)
        remain_positions = np.array(remain_positions)

        # damage differential
        A_hat_batch = []

        all_neighbor = Utils.calculate_khop_neighbour(fixed_positions, config_communication_range)
        # batch_size = max([len(hop) for hop in all_neighbor])-1

        for k in range(khop, khop+batch_size):
            A = Utils.make_khop_A_matrix(all_neighbor, fixed_positions, num_remain, k)
            # A = Utils.make_A_matrix(remain_positions, num_remain, config_communication_range)

            D = Utils.make_D_matrix(A, num_of_agents)
            L = D - A
            A_norm = np.linalg.norm(A, ord=np.inf)
            # print(A_norm)
            # A_norm = num_of_agents
            k0 = 1 / A_norm
            K = 0.5 * k0
            A_hat_khop = np.eye(num_of_agents) - K * L
            
            A_hat_khop = torch.FloatTensor(A_hat_khop).type(FloatTensor)
            A_hat_batch.append(A_hat_khop)

        A_hat = torch.block_diag(A_hat_batch[0])
        for k in range(1, batch_size):
            A_hat = torch.block_diag(A_hat, A_hat_batch[k])

        fixed_positions = torch.FloatTensor(fixed_positions).type(FloatTensor)
        fixed_positions_batch = torch.tile(fixed_positions, (batch_size,1))
        remain_positions = torch.FloatTensor(remain_positions).type(FloatTensor)
        remain_positions_batch = torch.tile(remain_positions, (batch_size,1))

        best_final_positions = 0
        best_loss = 1000000000000
        loss_ = 0

        best_final_k = 0
        best_final_epoch = 0

        loss_storage = []
        num_storage = []

        # print("---------------------------------------")
        # print("start training GCN ... ")
        # print("=======================================")
        for train_step in range(50):
            # if train_step == 200:
            #     torch.save(gcn_network, './Pretrained_model/model.pt')

            final_positions_list = gcn_network(fixed_positions_batch, A_hat, num_remain)
            
            if dimension == 3:
                final_positions_list = [0.5 * torch.Tensor(np.array([config_width, config_length, config_height])).type(FloatTensor) * p for p in final_positions_list]
            else:
                final_positions_list = [0.5 * torch.Tensor(np.array([config_width, config_length])).type(FloatTensor) * p for p in final_positions_list]

            loss_list = []
            num_list = []

            loss = []

            for final_positions in final_positions_list:
                try:
                    e_vals, num = Utils.check_number_of_clusters_torch(final_positions, config_communication_range)
                except:
                    final_positions_ = final_positions.cpu().data.numpy()
                    A = Utils.make_A_matrix(final_positions_, len(final_positions_), config_communication_range)
                    D = Utils.make_D_matrix(A, len(A))
                    L = D - A
                    flag, num = Utils.check_number_of_clusters(L, len(L))
                # print(num.cpu().data.numpy(), num_)
                num_list.append(num)

                loss_k = 5000*(num-1) + torch.max(torch.norm(final_positions-remain_positions, dim=1)) + torch.sum(torch.norm(final_positions-remain_positions, dim=1))/num_remain
                # loss_k = 5000*(num-1) + torch.sum(torch.norm(final_positions-remain_positions, dim=1))/num_remain
                loss_list.append(loss_k)
                # loss_step.append(torch.max(torch.norm(final_positions-remain_positions, dim=1)))
                loss.append(torch.sum(torch.norm(final_positions-remain_positions, dim=1))/num_remain)

            best_loss_k = min(loss_list)
            best_k = loss_list.index(best_loss_k)
            final_positions = final_positions_list[best_k]
            num = num_list[best_k]

            loss = torch.stack(loss_list)
            # print(loss)

            # initialization
            if train_step == 0:
                loss_weights = torch.ones_like(loss)
                # loss_weights = torch.tensor((1,1,10)).cuda()
                loss_weights = torch.nn.Parameter(loss_weights)
                T = loss_weights.sum().detach()
                loss_optimizer = Adam([loss_weights], lr=0.01)
                l0 = loss.detach()
                layer = gcn_network.gc8

            # compute the weighted loss
            weighted_loss = loss_weights @ loss

            if best_loss_k.cpu().data.numpy() < best_loss:
                best_loss = deepcopy(best_loss_k.cpu().data.numpy())
                best_final_positions = deepcopy(final_positions.cpu().data.numpy())
                best_final_k = best_k
                best_final_epoch = train_step

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

            # compute loss_weights and loss
            loss_ratio = loss.detach() / l0
            rt = loss_ratio / loss_ratio.mean()
            gw_avg = gw.mean().detach()
            constant = (gw_avg * rt ** alpha).detach()
            gradnorm_loss = torch.abs(gw - constant).sum()

            # backword
            loss_optimizer.zero_grad()
            gradnorm_loss.backward()

            # update
            optimizer.step()
            loss_optimizer.step()

            # renormalize loss_weights
            loss_weights = (loss_weights / loss_weights.sum() * T).detach()
            loss_weights = torch.nn.Parameter(loss_weights)
            loss_optimizer = torch.optim.Adam([loss_weights], lr=0.01)
            
            loss_ = best_loss_k.cpu().data.numpy()
            num_ = num.cpu().data.numpy() if torch.is_tensor(num) else num
            print(f"episode {train_step}, num {num_}, loss {loss_:.6f}", end='\r')

            num_storage.append(int(num_))
            loss_storage.append(loss_ % 5000)
            

        speed = np.zeros((config_num_of_agents, dimension))
        remain_positions_numpy = remain_positions.cpu().data.numpy()
        temp_max_distance = 0
        # print("=======================================")

        for i in range(num_remain):
            if np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i]) > 0:
                speed[remain_list[i]] = (best_final_positions[i] - remain_positions_numpy[i]) / np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i])
            if np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i]) > temp_max_distance:
                temp_max_distance = deepcopy(np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i]))

        # speed = speed / np.max(np.linalg.norm(speed, axis=1))

        max_time = temp_max_distance / config_constant_speed

        print(f"trained: max time {max_time}, best episode {best_final_epoch}, best k-hop {best_final_k+1}")
            
        if save_loss_curve:
            with open(f'./Logs/loss/loss_d{num_destructed-1}_setup.txt', 'a') as f:
                print(loss_storage, file=f)

            with open(f'./Logs/loss/num_d{num_destructed-1}_setup.txt', 'a') as f:
                print(num_storage, file=f)

        return deepcopy(speed), deepcopy(max_time), deepcopy(best_final_positions)

    