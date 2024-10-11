from copy import deepcopy

import Utils
from Configurations import *
from Environment import Environment

from Previous_Algorithm.CR_MGC import CR_MGC
from Previous_Algorithm.DEMD import DEMD
from Previous_Algorithm.GCN_2017 import GCN_2017
from Previous_Algorithm.Centering import centering_fly
from Previous_Algorithm.SIDR import SIDR
from Previous_Algorithm.HERO import HERO

from MDSG_Algorithm.MDSG_GC_batch import MDSG_GC_batch
from MDSG_Algorithm.MDSG_APF import mdsg_apf, mdsg_apf_khop


class Swarm:
    def __init__(self, algorithm_mode=1, use_pretrained=False, khop=3):
        self.initial_positions = deepcopy(config_initial_swarm_positions)
        self.remain_list = [i for i in range(config_num_of_agents)]
        self.remain_num = config_num_of_agents
        self.num_of_agents = config_num_of_agents
        self.max_destroy_num = config_maximum_destroy_num

        self.remain_positions = deepcopy(self.initial_positions)
        self.true_positions = deepcopy(self.initial_positions)

        self.database = [{"known_positions": deepcopy(self.initial_positions),
                          "existing_list": [i for i in range(config_num_of_agents)],
                          "connected": True,
                          "if_destroyed": False} for i in range(config_num_of_agents)]
        # 0 for CSDS, 1 for centering, 2 for SIDR, 3 for GCN_2017, 4 for CR-GCM, 5 for CR_GCM_N
        self.algorithm_mode = algorithm_mode

        self.if_once_gcn = False
        self.once_destroy_gcn_speed = np.zeros((self.num_of_agents, config_dimension))
        self.max_time = 0

        self.khop = khop
        self.demd = DEMD()
        self.mdsg_gc = MDSG_GC_batch(use_pretrained=True)
        
        self.cr_gcm = CR_MGC(use_meta=False)
        self.gcn_2017 = GCN_2017()

        self.hero = HERO(self.initial_positions)

        self.if_once_gcn_network = False
        self.once_destroy_gcn_network_speed = np.zeros((self.num_of_agents, config_dimension))

        self.best_final_positions = 0

        self.notice_destroy = False
        self.destination_positions = np.zeros((self.num_of_agents, config_dimension))
        self.inertia_counter = 0
        self.inertia = 100
        self.if_finish = [True for i in range(self.num_of_agents)]

    def destroy_happens(self, destroy_list, environment_positions):
        self.notice_destroy = True
        for destroy_index in destroy_list:
            self.remain_list.remove(destroy_index)
        self.true_positions = deepcopy(environment_positions)
        self.remain_num = len(self.remain_list)
        # self.csds.notice_destroy(deepcopy(destroy_list))

    def update_true_positions(self, environment_positions):
        self.true_positions = deepcopy(environment_positions)

    def reset(self, change_algorithm_mode=False, algorithm_mode=0):
        self.remain_list = [i for i in range(config_num_of_agents)]
        self.remain_num = config_num_of_agents
        self.database = [{"known_positions": deepcopy(self.initial_positions),
                          "existing_list": [i for i in range(config_num_of_agents)],
                          "connected": True,
                          "if_destroyed": False} for i in range(config_num_of_agents)]
        self.positions = []
        self.mean_positions = []
        self.target_positions = []
        self.max_time = 0

        if change_algorithm_mode:
            self.algorithm_mode = algorithm_mode

        self.if_once_gcn = False
        self.once_destroy_gcn_speed = np.zeros((self.num_of_agents, config_dimension))

        self.if_once_gcn_network = False
        self.once_destroy_gcn_network_speed = np.zeros((self.num_of_agents, config_dimension))

    def take_actions(self):
        """
        take actions with global information (GI)
        :return: unit speed vectors
        """
        actions = np.zeros((self.num_of_agents, config_dimension))
        max_time = 0
        self.make_remain_positions()
        flag, num_cluster = Utils.check_if_a_connected_graph(deepcopy(self.remain_positions), len(self.remain_list))
        if flag:
            # print("connected")
            return deepcopy(actions), max_time
        else:
            # HERO
            if self.algorithm_mode == 1:
                actions_hero = self.hero.hero(
                    Utils.difference_set([i for i in range(self.num_of_agents)], self.remain_list), self.true_positions)

                for i in self.remain_list:
                    actions[i] = 0.2 * centering_fly(self.true_positions, self.remain_list, i) + 0.8 * actions_hero[i]


            # centering
            elif self.algorithm_mode == 2:
                for i in self.remain_list:
                    actions[i] = centering_fly(self.true_positions, self.remain_list, i)


            # SIDR
            elif self.algorithm_mode == 3:
                actions = SIDR(self.true_positions, self.remain_list)


            # GCN
            elif self.algorithm_mode == 4:
                if self.if_once_gcn_network:
                    for i in range(len(self.remain_list)):
                        if np.linalg.norm(
                                self.true_positions[self.remain_list[i]] - self.best_final_positions[i]) >= 0.55:
                            actions[self.remain_list[i]] = deepcopy(
                                self.once_destroy_gcn_network_speed[self.remain_list[i]])
                        # else:
                        #     print("%d already finish" % self.remain_list[i])
                    max_time = deepcopy(self.max_time)
                else:
                    self.if_once_gcn_network = True
                    actions, max_time, best_final_positions = self.gcn_2017.cr_gcm_n(deepcopy(self.true_positions),
                                                                                     deepcopy(self.remain_list))
                    self.once_destroy_gcn_network_speed = deepcopy(actions)
                    self.best_final_positions = deepcopy(best_final_positions)
                    self.max_time = deepcopy(max_time)

            
            # CR-MGC
            elif self.algorithm_mode == 5:
                if self.if_once_gcn_network:
                    for i in range(len(self.remain_list)):
                        if np.linalg.norm(self.true_positions[self.remain_list[i]] - self.best_final_positions[i]) >= 0.55:
                            actions[self.remain_list[i]] = deepcopy(self.once_destroy_gcn_network_speed[self.remain_list[i]])

                    max_time = deepcopy(self.max_time)
                else:
                    self.if_once_gcn_network = True
                    actions, max_time, best_final_positions = self.cr_gcm.cr_gcm(deepcopy(self.true_positions),
                                                                                 deepcopy(self.remain_list))
                    self.once_destroy_gcn_network_speed = deepcopy(actions)
                    self.best_final_positions = deepcopy(best_final_positions)
                    self.max_time = deepcopy(max_time)


            # DEMD
            elif self.algorithm_mode == 6:
                if self.if_once_gcn_network:
                    for i in range(len(self.remain_list)):
                        d = np.linalg.norm(self.true_positions[self.remain_list[i]] - self.best_final_positions[i])
                        if d >= 1:
                            actions[self.remain_list[i]] = deepcopy(self.once_destroy_gcn_network_speed[self.remain_list[i]])
                        elif d > 0.0001:
                            actions[self.remain_list[i]] = deepcopy(self.best_final_positions[i] - self.true_positions[self.remain_list[i]])
                    max_time = deepcopy(self.max_time)
                else:
                    self.if_once_gcn_network = True
                    # actions, max_time, best_final_positions = self.demd.demd(deepcopy(self.true_positions), deepcopy(self.remain_list), self.khop)
                    actions, max_time, best_final_positions = self.demd.demd_adaptive(deepcopy(self.true_positions), deepcopy(self.remain_list))
                    self.once_destroy_gcn_network_speed = deepcopy(actions)
                    self.best_final_positions = deepcopy(best_final_positions)
                    self.max_time = deepcopy(max_time)
            
            
            # proposed MDSG-APF algorithm
            elif self.algorithm_mode == 7:
                if self.if_once_gcn_network:
                    for i in range(len(self.remain_list)):
                        actions[self.remain_list[i]] = deepcopy(self.once_destroy_gcn_network_speed[self.remain_list[i]])
                else:
                    self.if_once_gcn_network = True
                    # actions = mdsg_apf(deepcopy(self.true_positions), deepcopy(self.remain_list), config_dimension, config_communication_range, self.khop)
                    actions = mdsg_apf_khop(deepcopy(self.true_positions), deepcopy(self.remain_list), config_dimension, config_communication_range, 8)
                    self.once_destroy_gcn_network_speed = deepcopy(actions)
                    
            
            # proposed MDSG-GC algorithm
            elif self.algorithm_mode == 8:
                if self.if_once_gcn_network:
                    for i in range(len(self.remain_list)):
                        d = np.linalg.norm(self.true_positions[self.remain_list[i]] - self.best_final_positions[i])
                        if d >= 1:
                            actions[self.remain_list[i]] = deepcopy(self.once_destroy_gcn_network_speed[self.remain_list[i]])
                        elif d > 0.0001:
                            actions[self.remain_list[i]] = deepcopy(self.best_final_positions[i] - self.true_positions[self.remain_list[i]])

                    max_time = deepcopy(self.max_time)
                else:
                    self.if_once_gcn_network = True
                    actions, max_time, best_final_positions = self.mdsg_gc.mdsg_gc_batch(deepcopy(self.true_positions), deepcopy(self.remain_list))
                    self.once_destroy_gcn_network_speed = deepcopy(actions)
                    self.best_final_positions = deepcopy(best_final_positions)
                    self.max_time = deepcopy(max_time)

            else:
                print("No such algorithm")
        return deepcopy(actions), deepcopy(max_time)

    def make_remain_positions(self):
        self.remain_positions = []
        for i in self.remain_list:
            self.remain_positions.append(deepcopy(self.true_positions[i]))
        self.remain_positions = np.array(self.remain_positions)

    def check_if_finish(self, cluster_index):
        flag = True
        for i in range(len(cluster_index)):
            if not self.if_finish[self.remain_list[cluster_index[i]]]:
                flag = False
                break
        return flag


if __name__ == "__main__":
    np.random.seed(57)
    random.seed(77)

    environment = Environment()
    swarm = Swarm()

    environment_positions = environment.reset()
    destroy_num, destroy_list = environment.stochastic_destroy(mode=2, num_of_destroyed=100)

    swarm.destroy_happens(deepcopy(destroy_list), deepcopy(environment_positions))

    print(len(swarm.remain_list))
