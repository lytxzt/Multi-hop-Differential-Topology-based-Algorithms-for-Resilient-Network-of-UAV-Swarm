import matplotlib.animation as animation
import matplotlib.pyplot as plt
from copy import deepcopy

from Environment import Environment
from Swarm import Swarm
from Configurations import *
import Utils

# determine if draw the video
"""
Note: if true, it may take a little long time
"""
config_draw_video = True
show_degree = False

# determine if use pre-trained model
use_pretrained = True

"""
    algorithm mode: 1 for HERO
                    2 for CEN
                    3 for SIDR
                    4 for GCN-2017
                    5 for CR-MGC
                    6 for DEMD
                    7 for MDSG-APF (no GCN)
                    8 for MDSG-GC (best algorithm)
"""
# set this value to 7 to run the proposed algorithm
config_algorithm_mode = 6
algorithm_mode = {1: "HERO",
                  2: "CEN",
                  3: "SIDR",
                  4: "GCN_2017",
                  5: "CR-MGC",
                  6: "DEMD",
                  7: "MDSG-APF",
                  8: "MDSG-GC"}

print("CNS issue Starts...")
print("------------------------------")
print("Algorithm: %s" % (algorithm_mode[config_algorithm_mode]))

environment = Environment()
swarm = Swarm(algorithm_mode=config_algorithm_mode, use_pretrained=use_pretrained)
num_cluster_list = []

environment_positions = environment.reset()
swarm.reset()

# storage
storage_remain_list = []
storage_positions = []
# storage_destroy_positions = []
storage_connection_states = []
storage_remain_connectivity_matrix = []

# change the number of destructed UAVs
config_num_destructed_UAVs = 100  # should be in the range of [1, config_num_-2]

# change the seed to alternate the UED
np.random.seed(17)
random.seed(18)

# destruction
storage_remain_list.append(deepcopy(swarm.remain_list))
storage_positions.append(deepcopy(swarm.true_positions))
# storage_destroy_positions.append([])
storage_connection_states.append(True)
storage_remain_connectivity_matrix.append(deepcopy(Utils.make_A_matrix(swarm.true_positions, config_num_of_agents, config_communication_range)))

# flag if break the CCN
break_CCN_flag = True
# num of connected steps
num_connected_steps = 0
count_connect = 0

for step in range(310):
    # destroy at time step 0
    if step == 0:
        print("=======================================")
        print("destroy %d -- mode %d num %d " % (0, 2, config_num_destructed_UAVs))
        destroy_num, destroy_list = environment.stochastic_destroy(mode=2, num_of_destroyed=config_num_destructed_UAVs)
        # real_destroy_list = [64, 83, 50, 61, 49, 29, 74, 121, 142, 156, 60, 37, 191, 161, 169, 128, 0, 71, 7, 127, 44, 48, 129, 99, 46, 84, 82, 30, 22, 141, 66, 136, 120, 52, 57, 195, 18, 20, 51, 143, 112, 152, 137, 19, 25, 54, 174, 171, 24, 147]
        # destroy_num, destroy_list = environment.stochastic_destroy(mode=4, real_destroy_list=real_destroy_list)
        print("destroy %d nodes \ndestroy index list :" % config_num_destructed_UAVs)
        print(destroy_list)
        swarm.destroy_happens(deepcopy(destroy_list), deepcopy(environment_positions))
        
        initial_remain_positions = []
        destroy_positions = []
        for i in range(config_num_of_agents):
            if i in destroy_list:
                destroy_positions.append(deepcopy(config_initial_swarm_positions[i]))
            else:
                initial_remain_positions.append(deepcopy(config_initial_swarm_positions[i]))
        initial_remain_positions = np.array(initial_remain_positions)
        destroy_positions = np.array(destroy_positions)
        A = Utils.make_A_matrix(initial_remain_positions, config_num_of_agents - config_num_destructed_UAVs,
                                config_communication_range)
        num_cluster = environment.check_the_clusters()

        # check if the UED break the CCN of the USNET
        if num_cluster == 1:
            print("---------------------------------------")
            print("This damage case does not split the USNET!")
            print("Please change the random seed or the number of destructed UAVs!")
            print("Algorithm Ends")
            print("---------------------------------------")
            break_CCN_flag = False
            break
        
        ## draw Fig11.a
        # positions_with_clusters = Utils.split_the_positions_into_clusters(initial_remain_positions, num_cluster, A)
        
        # fig = plt.figure()
        # plt.axes(xlim=(0, 1000), ylim=(0, 1000))
        # text_pos = [[700,800], [630,250], [805,610], [190,550], [450,120], [20,20], [700,30]]
        # arrow_start = [[625,660], [710,190], [940,585], [175,440], [500,80], [50,80], [860,46]]
        # arrow_end = [[760,790], [690,240], [865,600], [250,540], [510,110], [80,50], [825,50]]

        # for i, pos in enumerate(positions_with_clusters):
        #     x, y = np.array(pos)[:,0], np.array(pos)[:,1]
        #     plt.scatter(x, y, s=30, zorder=4)
        #     # plt.text(np.mean(pos, axis=0)[0], np.mean(pos, axis=0)[1], f'{i}')
        #     plt.text(text_pos[i][0], text_pos[i][1], f'sub-net{i+1}', c='black')
        #     dx, dy = arrow_start[i][0] - arrow_end[i][0], arrow_start[i][1] - arrow_end[i][1]
        #     plt.arrow(arrow_end[i][0], arrow_end[i][1], dx, dy, width=2, head_width=16, ec='black', fc='black', zorder=3)

        # for pos in destroy_positions:
        #     plt.scatter(pos[0], pos[1], s=15, c='black', zorder=2)

        # for p1 in initial_remain_positions:
        #     for p2 in initial_remain_positions:
        #         if np.linalg.norm(p1-p2) <= config_communication_range:
        #             x, y = [p1[0], p2[0]], [p1[1], p2[1]]
        #             plt.plot(x, y, c='lightsteelblue', linewidth=2, zorder=1)

        # plt.scatter(pos[0], pos[1], s=15, c='black', zorder=2, label='destructed UAVs')
        # plt.plot(x, y, c='lightsteelblue', linewidth=2, zorder=1, label='communication links')

        # plt.legend(loc='upper left')
        # plt.xlabel('Ground X', fontdict={'family':'serif', 'size':14})
        # plt.ylabel('Ground Y', fontdict={'family':'serif', 'size':14})
        # plt.savefig('./case1.png', dpi=600, bbox_inches='tight')
        # plt.show()

        print("=======================================")

    actions, max_time = swarm.take_actions()
    environment_next_positions = environment.next_state(deepcopy(actions))
    swarm.update_true_positions(environment_next_positions)

    temp_cluster = environment.check_the_clusters()
    num_cluster_list.append(temp_cluster)
    print("---------------------------------------")
    if temp_cluster == 1:
        print(f"step {step} ---num of sub-nets {environment.check_the_clusters()} -- connected")
        count_connect += 1
        if count_connect > 15:
            break
    else:
        num_connected_steps += 1
        print(f"step {step} ---num of sub-nets {environment.check_the_clusters()} -- disconnected --max time {max_time}")

    storage_remain_list.append(deepcopy(swarm.remain_list))

    storage_positions.append(deepcopy(environment_next_positions))
    
    if environment.check_the_clusters() == 1:
        storage_connection_states.append(True)
    else:
        storage_connection_states.append(False)
    remain_positions = []
    for i in swarm.remain_list:
        remain_positions.append(deepcopy(environment_next_positions[i]))
    remain_positions = np.array(remain_positions)
    storage_remain_connectivity_matrix.append(
        deepcopy(Utils.make_A_matrix(remain_positions, len(swarm.remain_list), config_communication_range)))

    # update
    environment.update()
    environment_positions = deepcopy(environment_next_positions)


if break_CCN_flag:
    print("=======================================")
    print("plotting trajectories of nodes...")
    final_positions = []
    for i in swarm.remain_list:
        final_positions.append(deepcopy(storage_positions[-1][i]))
    final_positions = np.array(final_positions)

    def update(frame):
        # ax = Axes3D(fig)
        plt.clf()
        ax = plt.axes(xlim=(0, 1000), ylim=(0, 1000))

        for i in range(len(storage_remain_list[frame])):
            for j in range(i, len(storage_remain_list[frame])):
                if storage_remain_connectivity_matrix[frame][i, j] == 1:
                    x = [storage_positions[frame][storage_remain_list[frame][i], 0],
                         storage_positions[frame][storage_remain_list[frame][j], 0]]
                    y = [storage_positions[frame][storage_remain_list[frame][i], 1],
                         storage_positions[frame][storage_remain_list[frame][j], 1]]
                    # z = [storage_positions[frame][storage_remain_list[frame][i], 2],
                    #      storage_positions[frame][storage_remain_list[frame][j], 2]]
                    ax.plot(x, y, c='lightsteelblue', linewidth=2, zorder=1)

        for i in range(config_num_of_agents):
            if i in storage_remain_list[frame]:
                x = [storage_positions[frame][i, 0],
                     config_initial_swarm_positions[i, 0]]
                y = [storage_positions[frame][i, 1],
                     config_initial_swarm_positions[i, 1]]
                # z = [storage_positions[frame][i, 2],
                #      config_initial_swarm_positions[i, 2]]
                ax.plot(x, y, c='blue', zorder=2)
                
        for i in range(config_num_of_agents):
            if i in storage_remain_list[frame]:
                ax.scatter(storage_positions[frame][i, 0], storage_positions[frame][i, 1],
                        #    storage_positions[frame][i, 2],
                           s=30, c='g', zorder=4)
            else:
                if frame <= 10:
                    ax.scatter(storage_positions[frame][i, 0], storage_positions[frame][i, 1], s=15, c='black', zorder=3)
                    
        ax.text(5, 100, 'time steps = %d' % (frame), c='b', zorder=5)
        ax.text(5, 60, 'number_of_clusters = %d' % num_cluster_list[frame], c='b', zorder=5)

        if storage_connection_states[frame]:
            ax.text(5, 20, 'Connected...', c='g')
        else:
            ax.text(5, 20, 'Unconnected...', c='r')
        # plt.set_zlabel('Height', fontdict={'size': 15, 'color': 'black'})
        # ax.ylabel('Ground Y', fontdict={'size': 15, 'color': 'black'})
        # ax.xlabel('Ground X', fontdict={'size': 15, 'color': 'black'})
        ax.set_xticks([])
        ax.set_yticks([])

        print("finish frame %d ..." % frame)

    if config_draw_video:
        ## draw video of recovery process
        print("=======================================")
        print("Plotting the dynamic trajectory...")
        fig = plt.figure()
        frame = np.linspace(0, num_connected_steps + 10, num_connected_steps + 11).astype(int)
        ani = animation.FuncAnimation(fig, update, frames=frame, interval=90, repeat_delay=10)
        ani.save(f"Figs/gif/CNS_d{config_num_destructed_UAVs}_{algorithm_mode[config_algorithm_mode]}.gif", writer='pillow', bitrate=2048, dpi=500)
        plt.show()

    elif config_algorithm_mode in [7, 8]:
        ## draw Fig11.c and Fig11.d
        fig = plt.figure()
        frame = num_connected_steps
        # update(num_connected_steps)
        ax = plt.axes(xlim=(0, 1000), ylim=(0, 1000))

        for i in range(len(storage_remain_list[frame])):
            for j in range(i, len(storage_remain_list[frame])):
                if storage_remain_connectivity_matrix[frame][i, j] == 1:
                    x = [storage_positions[frame][storage_remain_list[frame][i], 0],
                         storage_positions[frame][storage_remain_list[frame][j], 0]]
                    y = [storage_positions[frame][storage_remain_list[frame][i], 1],
                         storage_positions[frame][storage_remain_list[frame][j], 1]]
                    # z = [storage_positions[frame][storage_remain_list[frame][i], 2],
                    #      storage_positions[frame][storage_remain_list[frame][j], 2]]
                    plt.plot(x, y, c='lightsteelblue', linewidth=2, zorder=1)

        for i in range(config_num_of_agents):
            if i in storage_remain_list[frame]:
                x = [storage_positions[frame][i, 0],
                     config_initial_swarm_positions[i, 0]]
                y = [storage_positions[frame][i, 1],
                     config_initial_swarm_positions[i, 1]]
                # z = [storage_positions[frame][i, 2],
                #      config_initial_swarm_positions[i, 2]]
                plt.plot(x, y, c='b', zorder=3)
                
        for i in range(config_num_of_agents):
            if i in storage_remain_list[frame]:
                plt.scatter(storage_positions[frame][i, 0], storage_positions[frame][i, 1], s=30, c='green', zorder=4)
                plt.scatter(storage_positions[0][i, 0], storage_positions[0][i, 1], s=15, c='lightgreen', zorder=2)

        plt.scatter(storage_positions[frame][2, 0], storage_positions[frame][2, 1], s=30, c='green', zorder=4, label='final UAVs')
        plt.scatter(storage_positions[0][2, 0], storage_positions[0][2, 1], s=15, c='lightgreen', zorder=2, label='original UAVs')
        plt.plot(x, y, c='b', zorder=3, label='trajectories')

        plt.xlabel('Ground X', fontdict={'family':'serif', 'size':14})
        plt.ylabel('Ground Y', fontdict={'family':'serif', 'size':14})
        plt.legend()

        # plt.savefig('./case4.png', dpi=600, bbox_inches='tight')
        plt.show()

    with open(f'./Logs/case/{algorithm_mode[config_algorithm_mode]}.txt', 'w') as f:
        print(num_cluster_list, file=f)