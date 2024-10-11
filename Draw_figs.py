import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg
from copy import deepcopy

colors = list(mcolors.TABLEAU_COLORS.keys())
plt.rcParams['axes.axisbelow'] = True


# draw Fig.7
def draw_khop():
    khop = [i+1 for i in range(8)]
    num_destroyed = [i for i in range(10, 200, 10)]
    # num_destroyed = [50, 100, 150]

    khop_step = []
    khop_count = [0 for _ in range(8)]

    methods = "MDSG-APF"
    khop_labels = [f'$N_D$={i}' for i in num_destroyed]

    for dnum in num_destroyed:
        khop_step_dnum = []
        
        with open(f'./Logs/khop/{methods}_d{dnum}.txt', 'r') as f:
            data = f.read().split('\n')

        step_list = [d.replace(' ','').strip('[').strip(']') for d in data if len(d) > 0]
        for step in step_list:
            s = [float(s)/10 for s in step.split(',')]
            khop_step_dnum.append(s)

            index = np.argwhere(s == np.min(s)).flatten()
            for i in index:
                khop_count[i] += 1/len(index)

        # print(khop_step_dnum)
        khop_step.append(np.mean(khop_step_dnum, axis=0))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.bar(khop, khop_count, label='distribution of $k^*$', width=0.5)
    ax1.set_xlabel('Value of $k$', fontdict={'family':'serif', 'size':14})
    ax1.set_ylabel('Distribution of $k^*$', fontdict={'family':'serif', 'size':14})

    ax2 = ax1.twinx()
    ax2.plot(khop, np.mean(khop_step, axis=0), c='r', marker='^', label='recovery time $T_{rc}$', linewidth=2)
    ax2.set_ylabel('Average recovery time $T_{rc}$ /s', fontdict={'family':'serif', 'size':14})

    # plt.xlim(8,192)
    # plt.ylim(-2,54)
    ax1.grid(axis='y', linestyle='--')
    fig.legend(loc='upper right',bbox_to_anchor=(0.76,0.86))

    plt.gcf().subplots_adjust(right=0.85)

    plt.savefig('./Figs/fig7.png', dpi=600, bbox_inches='tight')
    plt.show()


# draw Fig.8
def draw_batch():
    plt.figure()
    # tips = sns.load_dataset('tips')
    # sns.boxplot(x='day', y='tip', hue='sex', data=tips)
    # print(tips)
    methods = ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'batch']
    methods_label = ['k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7', 'k=8', 'k=9', 'batch\nprocessing']
    # methods = ['k1', 'k2']
    variable = [50, 100, 150]

    # positions = [[i for i in range(1, 4*len(methods), 4)], [i for i in range(2, 4*len(methods), 4)], [i for i in range(3, 4*len(methods), 4)]]
    positions = [[i for i in range(k, (len(variable)+1)*len(methods), len(variable)+1)] for k in range(1, len(variable)+1)]
    position_tick = [i for i in range(2, (len(variable)+1)*len(methods), len(variable)+1)]

    df = {'d50':[], 'd100':[], 'd150':[]}

    for dnum in variable:
        step_list = []

        for m in methods:
            with open(f'./Logs/batch/MDSG-GC_d{dnum}_{m}.txt', 'r') as f:
                    data = f.read().split('\n')

            step = data[4].replace(' ','').strip('[').strip(']')
            step = [min(float(s)/10, 49.9) for s in step.split(',')]
            step_list.append(step)

        df[f'd{dnum}'] = deepcopy(step_list)

    # print(df)
    handles = []
    for i, var in enumerate(variable):
        bp = plt.boxplot(df[f'd{var}'], positions=positions[i], patch_artist=True)
        handles.append(bp['boxes'][0])

        for patch in bp['boxes']:
            patch.set_facecolor(mcolors.TABLEAU_COLORS[colors[i+2]])

    plt.xticks(position_tick, methods_label)
    plt.legend(handles=handles, labels=[f'$N_D=${var}' for var in variable], loc='upper right')
    plt.grid(axis='y', linestyle='--')
    
    # plt.xlabel('Number of destroyed UAVs $N_D$', fontdict={'family':'serif', 'size':14})
    plt.ylabel('Recovery time $T_{rc}$ /s', fontdict={'family':'serif', 'size':14})
    plt.savefig('./Figs/fig8.png', dpi=600, bbox_inches='tight')
    plt.show()


# draw Fig.9
def draw_loss_curve():

    file_loss = ['loss_d50_setup', 'loss_d100_setup', 'loss_d150_setup', 'loss_d50', 'loss_d100', 'loss_d150']
    loss_label = ['$N_D=50$', '$N_D=100$', '$N_D=150$', '$N_D=50$', '$N_D=100$', '$N_D=150$']

    loss_curve_list = []
    loss_std_list = []
    step_range = range(1000)

    for k in range(len(file_loss)):
        with open(f'./Logs/loss/{file_loss[k]}.txt', 'r') as f:
            data = f.read()

        data = data.split('\n')[:-1]

        for i in range(len(data)):
            data[i] = [float(d) for d in data[i].replace(' ','').strip('[').strip(']').split(',')]

        loss = np.array(data)
        # print(loss)
        # loss = loss[:5]
        loss_mean = np.mean(loss, axis=0)
        loss_curve_list.append(loss_mean)

        loss_std = 1.96*np.std(loss, ddof=1)/np.sqrt(len(loss))
        loss_std_list.append(loss_std)


    plt.figure()
    for k in range(len(file_loss)):
        plt.plot(step_range, loss_curve_list[k], label=loss_label[k], c=colors[k], linewidth=2)
        plt.fill_between(step_range, loss_curve_list[k]-loss_std_list[k], loss_curve_list[k]+loss_std_list[k], facecolor=colors[k], alpha=0.2)

    plt.xlim(0,100)
    # plt.ylim(100,1100)
    plt.grid(linestyle='--')
    plt.xlabel('Training Episode', fontdict={'family':'serif', 'size':14})
    plt.ylabel('Loss Curve', fontdict={'family':'serif', 'size':14})
    plt.legend(loc='upper right', ncol=2, title='   pre-trained              random    \n  initialization           initialization  ')
    plt.savefig('./Figs/fig9.png', dpi=600, bbox_inches='tight')
    plt.show()


# draw Fig.10a
def draw_method_figs():

    # num_destroyed = [25, 50, 75, 100, 125, 150, 175]
    num_destroyed = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]

    step_methods = []
    step_methods_std = []

    marklist = ['o', '^', 's', 'd', 'h', 'v', '>', '8']
    methods = ["MDSG-GC", "MDSG-APF", "CEN", "HERO", "SIDR", "GCN_2017", "CR-MGC", "DEMD"]
    method_labels = ["MDSG-GC", "MDSG-APF", "centering", "HERO", "SIDR", "GCN", "CR-MGC", "DEMD"]

    for m in methods:
        step_m = []
        step_std = []
        
        for dnum in num_destroyed:
            try:
                with open(f'./Logs/damage/d{dnum}/{m}_d{dnum}.txt', 'r') as f:
                    data = f.read().split('\n')

                # print(data[7].replace(' ',''))
                step_list = data[4].replace(' ','').strip('[').strip(']')
                step_list = [min(float(s)/10, 49.9) for s in step_list.split(',')]

                step_m.append(float(data[7].replace(' ',''))/10)
                step_std.append(1.96*np.std(step_list, ddof=1)/np.sqrt(len(step_list)))
            except:
                step_m.append(49.9)
                step_std.append(0)

        # print(step_std)
        step_methods.append(step_m)
        step_methods_std.append(step_std)

    plt.figure()
    for i in reversed(range(len(step_methods))):
        # plt.fill_between(num_destroyed, np.array(step_methods[i])-np.array(step_methods_std[i]), np.array(step_methods[i])+np.array(step_methods_std[i]), color=mcolors.TABLEAU_COLORS[colors[i]], alpha=0.2)
        plt.fill_between(num_destroyed, [max(s - s_std, 0) for s, s_std in zip(step_methods[i], step_methods_std[i])], [min(s + s_std, 49.9) for s, s_std in zip(step_methods[i], step_methods_std[i])], color=mcolors.TABLEAU_COLORS[colors[i]], alpha=0.2)

    for i, s in enumerate(step_methods):
        plt.plot(num_destroyed, s, c=mcolors.TABLEAU_COLORS[colors[i]],marker=marklist[i],label=method_labels[i], linewidth=2)

    plt.xlim(8,192)
    plt.ylim(-2,54)
    plt.grid(linestyle='--')
    plt.xlabel('Number of destroyed UAVs $N_D$', fontdict={'family':'serif', 'size':14})
    plt.ylabel('Average recovery time $T_{rc}$ /s', fontdict={'family':'serif', 'size':14})
    plt.legend(loc='upper left')
    # plt.plot(num_destroyed, step_methods[2], c=mcolors.TABLEAU_COLORS[colors[2]],marker=marklist[2])
    plt.plot(num_destroyed, step_methods[1], c=mcolors.TABLEAU_COLORS[colors[1]],marker=marklist[1], linewidth=2)
    plt.plot(num_destroyed, step_methods[0], c=mcolors.TABLEAU_COLORS[colors[0]],marker=marklist[0], linewidth=2)

    plt.savefig('./Figs/fig10a.png', dpi=600, bbox_inches='tight')
    plt.show()

    ratio = []
    for i in range(len(num_destroyed)):
        ratio.append((step_methods[7][i] - step_methods[1][i])/step_methods[7][i])

    print(ratio, sum(ratio)/len(num_destroyed))


# draw Fig.10b
def draw_spatial_coverage():

    # num_destroyed = [25, 50, 75, 100, 125, 150, 175]
    num_destroyed = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]

    coverage_methods = []
    coverage_methods_std = []

    # coverage_methods = [[0.5790114182692307, 0.54329453125, 0.47705390624999994, 0.4067041666666667, 0.31975442708333335, 0.24608235677083334, 0.15416647135416667], [0.551693835136218, 0.5039954427083333, 0.45265026041666667, 0.36421516927083336, 0.27528483072916665, 0.20392389322916663, 0.13745338541666668], [0.5472358273237179, 0.5137605794270833, 0.42624472656250006, 0.3349951171875001, 0.23151835937499998, 0.1789016927083333, 0.10261953124999998]]
    # scaled
    # coverage_methods = [[0.5790114182692307, 0.54329453125, 0.47705390624999994, 0.4067041666666667, 0.31975442708333335, 0.24608235677083334, 0.15416647135416667], [0.551693835136218, 0.5039954427083333, 0.45265026041666667, 0.36421516927083336, 0.27528483072916665, 0.20392389322916663, 0.13745338541666668], [0.5873599008413461, 0.5707125, 0.5360072916666666, 0.4860467447916667, 0.39812076822916664, 0.2877386067708333, 0.17947519531249997]]
    # coverage_methods = [[0.5968694911858974, 0.5925361328125, 0.5809194010416667, 0.5435434895833333, 0.45511796875, 0.3193609375, 0.1830646484375], [0.5790114182692307, 0.54329453125, 0.47705390624999994, 0.4067041666666667, 0.31975442708333335, 0.24608235677083334, 0.15416647135416667], [0.551693835136218, 0.5039954427083333, 0.45265026041666667, 0.36421516927083336, 0.27528483072916665, 0.20392389322916663, 0.13745338541666668], [0.5472358273237179, 0.5137605794270833, 0.42624472656250006, 0.34188515625, 0.23151835937499998, 0.17538046875, 0.10261953124999998], [0.5873599008413461, 0.5707125, 0.5360072916666666, 0.4860467447916667, 0.39812076822916664, 0.2877386067708333, 0.17947519531249997]]
    # coverage_methods = [[0.5983902037377451, 0.5976621547965116, 0.5968040364583334, 0.5959264322916668, 0.5943787760416668, 0.5924305989583334, 0.5851134114583334, 0.5772955078124999, 0.570451953125, 0.554644140625, 0.5338967447916667, 0.5003774739583333, 0.440012890625], [0.5967989813112746, 0.5957642169331395, 0.5915326822916667, 0.5891501302083334, 0.5797401692708334, 0.5751291666666668, 0.5582727213541666, 0.5415691406250001, 0.5158234375, 0.4898411458333334, 0.4528272135416666, 0.4151361328125, 0.3673423828125], [0.5957640165441177, 0.59012574188469, 0.5782962890625, 0.5684613932291667, 0.5511035156249999, 0.5230518229166667, 0.5044772786458334, 0.4802637369791667, 0.44317441406249997, 0.40146946614583334, 0.3644350911458333, 0.33322597656249997, 0.28766243489583326], [0.5920214843750001, 0.5881481649709303, 0.5580357421875, 0.5394173177083332, 0.5403128906250001, 0.5121793619791667, 0.4902163411458333, 0.43920748697916673, 0.42078157552083334, 0.38176901041666667, 0.3311368489583333, 0.29600944010416663, 0.25664322916666665], [0.5742078354779412, 0.5636737675629845, 0.5404018880208333, 0.5475331380208333, 0.5159981119791667, 0.49667727864583333, 0.4658399088541667, 0.45294798177083323, 0.4053949869791667, 0.34687076822916674, 0.30522858072916664, 0.2649919921875, 0.2236537109375]]
    # coverage_methods = [[967.5024767801857, 1020.0100775193797, 1078.46, 1144.17875, 1217.2877333333336, 1299.962, 1382.6680000000001, 1477.8765000000003, 1593.1167272727273, 1703.8668], [964.9297213622291, 1016.770930232558, 1068.9343529411767, 1131.1682500000002, 1187.3078666666668, 1261.9977142857142, 1319.2413846153845, 1386.417, 1440.5541818181819, 1440.3568], [963.2563467492259, 1007.1479328165376, 1045.015411764706, 1091.445875, 1128.6599999999999, 1147.7251428571428, 1192.1186153846154, 1229.4751666666666, 1237.6652727272726, 1233.3142], [957.2052631578947, 1003.7728682170542, 1008.403411764706, 1035.6812499999999, 1106.5608000000002, 1123.8678571428572, 1158.4189230769232, 1124.3711666666668, 1175.1281818181817, 1172.7944], [928.4034055727554, 962.00322997416, 976.538, 1051.263625, 1056.7641333333333, 1089.851857142857, 1100.8155384615386, 1159.5468333333333, 1132.1576363636366, 1065.587]]
    coverage_methods = [[0.9871402606376765, 0.9771705650808795, 0.9665405774321641, 0.9632182743216411, 0.9495224602911978, 0.9431990403706153, 0.9074623593646591, 0.8847019771674387, 0.8646366230972865, 0.8269932991396426, 0.7830493050959629, 0.7252132279947054, 0.6333607710125744, 0.5528778540701522, 0.4817411896095301, 0.3911691346790204, 0.30570586532097943, 0.24395433487756446, 0.1653644109861019], [0.9714100274457894, 0.9647603619965215, 0.9393530774321639, 0.9345551373262737, 0.9013763649900726, 0.890968108868299, 0.8544889559894108, 0.8107023494374586, 0.761996401389808, 0.7098436465916612, 0.6526422898742554, 0.5979889146260754, 0.5300883107213765, 0.45416177200529445, 0.4138782263401721, 0.3438922898742554, 0.2824071393117141, 0.22980166280608863, 0.15788757445400395], [0.9059835714563786, 0.8856048858756712, 0.8342860688285902, 0.8477827183984116, 0.7823039377895431, 0.7529134265387161, 0.7003571724023824, 0.6717879301786895, 0.5994525562541362, 0.5058758686300462, 0.4433258189940436, 0.38552903706154856, 0.3272003226340172, 0.3016046906022501, 0.25274673229649236, 0.21181192091330242, 0.1700823130377233, 0.1402953342157511, 0.1130906684315023], [0.9545131048779536, 0.9367814322872577, 0.8631918017868959, 0.8301327763070813, 0.828414336532098, 0.776165205162144, 0.7337140138980806, 0.6449311300463267, 0.6180329665784249, 0.5537686135009926, 0.47980207643944406, 0.429423395102581, 0.3743561796823295, 0.34711035737921897, 0.30579686465916606, 0.2506090751158173, 0.22039109033752477, 0.18017496690933155, 0.13646426207809395], [0.9623824794643203, 0.9323341522632477, 0.9036149487094639, 0.8786343894771674, 0.843946475843812, 0.7830917025148907, 0.7477179847782923, 0.7046461366644604, 0.6474977250165453, 0.5824030029781601, 0.5291241313699536, 0.4829181833223031, 0.4181200363997351, 0.37784889973527463, 0.3379121856386498, 0.28597927696889475, 0.24378805426869615, 0.2067397418927862, 0.1414961118464593]]
    coverage_methods_std = [[0.005164016935172642, 0.006280623147680463, 0.006349164613790299, 0.007037124437072787, 0.00778364171335366, 0.008006848821802047, 0.011482439628413967, 0.010769280145376326, 0.010680714133991945, 0.0115880177425518, 0.011562086753975, 0.015499752619390307, 0.018263029875359358, 0.02011443882565077, 0.019971947877466572, 0.017024216945770124, 0.015708242743491727, 0.00916888193361664, 0.007449639073074311], [0.012071340352742863, 0.00900042365037345, 0.01211444014990316, 0.014825775790554975, 0.020234959431160795, 0.0226474705205137, 0.030545807617139438, 0.030119152088311243, 0.032789429870032916, 0.023533945890900045, 0.023816684403525752, 0.023113108642492136, 0.023088513417324923, 0.01751067425257052, 0.01882494270803785, 0.01633170734394814, 0.013765080229907335, 0.007205807162169809, 0.00464466101052132], [0.052566956021553826, 0.04122931645896945, 0.04358943564797798, 0.04155485361042807, 0.04836445208107116, 0.0586140009765723, 0.06470660722025666, 0.05947320303122341, 0.06435666272505393, 0.05404577161445782, 0.045292454772044986, 0.04574169291278129, 0.03934527420997144, 0.03565263874957842, 0.02960817491228032, 0.02441588760806246, 0.014997221993624455, 0.00870894586880823, 0.005062800484226115], [0.028140383786482166, 0.021733873918608225, 0.03690258174335679, 0.0441331493345681, 0.043954593380992335, 0.048654435990514, 0.051441591696489083, 0.05119624869638891, 0.05393334489447707, 0.04126567704448232, 0.03690027183950489, 0.02946953692100325, 0.02341486943060018, 0.019444355682434116, 0.01965314172270739, 0.014903471296361118, 0.014845932558468063, 0.01206918756527795, 0.007266287090729348], [0.016600864539432242, 0.01726773389668875, 0.023384422993803552, 0.027503187057832435, 0.03586704816236548, 0.03407020905226816, 0.03460339454159637, 0.03242595020416412, 0.036443854392037636, 0.033148873838583566, 0.029926668069118402, 0.024407605997728322, 0.01992317567381207, 0.021233210716307672, 0.016303944071905403, 0.014916742425670631, 0.018312415548373637, 0.012005107991926807, 0.006923392813456899]]

    marklist = ['o', '^', 's', 'd', 'v']
    methods = ["MDSG-GC", "MDSG-APF", "CEN", "CR-MGC", "DEMD"]
    method_labels = ["MDSG-GC", "MDSG-APF", "centering", "CR-MGC", "DEMD"]

    # config_initial_swarm_positions = pd.read_excel("Configurations/swarm_positions_200.xlsx")
    # config_initial_swarm_positions = config_initial_swarm_positions.values[:, 1:3]
    # config_initial_swarm_positions = np.array(config_initial_swarm_positions, dtype=np.float64)

    # plt.figure()
    # for (x,y) in config_initial_swarm_positions.tolist():
    #     plt.gcf().gca().add_artist(plt.Circle((x,y), 120, color='#000000'))

    # plt.xlim(0,1500)
    # plt.ylim(0,1500)

    # canvas = FigureCanvasAgg(plt.gcf())
    # canvas.draw()

    # w, h = canvas.get_width_height()

    # buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)

    # buf.shape = (w,h,4)
    # buf = np.roll(buf, 3, axis=2)
    # # print(np.sum(buf[:,:,1]==0)/(w*h))

    # plt.close()

    # area = np.sum(buf[:,:,1]==0)/(w*h)
    # print(area)
    area = 0.3147916666666667

    if len(coverage_methods) == 0:
        for m in methods:
            coverage_m = []
            coverage_m_std = []
            for dnum in num_destroyed:
                with open(f'./Logs/damage/d{dnum}/{m}_d{dnum}.txt', 'r') as f:
                    data = f.read().split('\n')

                print(m, dnum)
                coverage_d = []

                for col in range(15, len(data), 5):
                    pos = eval(data[col].replace('array', 'np.array'))

                    plt.figure()
                    for (x,y) in pos:
                        plt.gcf().gca().add_artist(plt.Circle((x,y), 120, color='#000000'))

                    plt.xlim(0,1500)
                    plt.ylim(0,1500)

                    canvas = FigureCanvasAgg(plt.gcf())
                    canvas.draw()

                    w, h = canvas.get_width_height()

                    buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)

                    buf.shape = (w,h,4)
                    buf = np.roll(buf, 3, axis=2)
                    # print(np.sum(buf[:,:,1]==0)/(w*h))

                    plt.close()

                    coverage_d.append(np.sum(buf[:,:,1]==0)/(w*h*area))
                    # coverage_d.append(np.sum(buf[:,:,1]==0)/(200-dnum))

                coverage_m.append(np.mean(coverage_d))
                coverage_m_std.append(1.96*np.std(coverage_d, ddof=1)/np.sqrt(len(coverage_d)))

            # print(coverage_m)
            coverage_methods.append(coverage_m)
            coverage_methods_std.append(coverage_m_std)

        print(coverage_methods)
        print(coverage_methods_std)
  
    plt.figure()
    for i in reversed(range(len(coverage_methods))):
        plt.fill_between(num_destroyed, [max(s - s_std, 0) for s, s_std in zip(coverage_methods[i], coverage_methods_std[i])], [min(s + s_std, 49.9) for s, s_std in zip(coverage_methods[i], coverage_methods_std[i])], color=mcolors.TABLEAU_COLORS[colors[i]], alpha=0.2)

    for i, s in enumerate(coverage_methods):
        plt.plot(num_destroyed, s, c=mcolors.TABLEAU_COLORS[colors[i]],marker=marklist[i],label=method_labels[i], linewidth=2)

    plt.xlim(8,192)
    plt.ylim(0.05,1.05)
    plt.grid(linestyle='--')
    plt.xlabel('Number of destroyed UAVs $N_D$', fontdict={'family':'serif', 'size':14})
    plt.ylabel('Average spatial coverage ratio', fontdict={'family':'serif', 'size':14})
    plt.plot(num_destroyed, coverage_methods[1], c=mcolors.TABLEAU_COLORS[colors[1]], marker=marklist[1], linewidth=2)
    plt.plot(num_destroyed, coverage_methods[0], c=mcolors.TABLEAU_COLORS[colors[0]], marker=marklist[0], linewidth=2)
    plt.legend(loc='upper right')

    plt.savefig('./Figs/fig10b.png', dpi=600, bbox_inches='tight')
    plt.show()

    ratio = []
    for i in range(len(num_destroyed)):
        ratio.append((coverage_methods[4][i] - coverage_methods[0][i])/coverage_methods[4][i])

    print(ratio, np.mean(ratio))


# draw Fig.10c and Fig.10d
def draw_degree_distribution():

    num_destroyed = [100, 150]
    # num_destroyed = [25, 50, 75, 100, 125, 150, 175]

    methods = ["MDSG-GC", "MDSG-APF", "CEN", "CR-MGC", "DEMD"]
    method_labels = ["MDSG-GC", "MDSG-APF", "centering", "CR-MGC", "DEMD"]

    for dnum in num_destroyed:
        drange = range(200-dnum)
        
        plt.figure()
        for i, m in enumerate(methods):
            with open(f'./Logs/damage/d{dnum}/{m}_d{dnum}.txt', 'r') as f:
                data = f.read()

            data = data.split('\n')
            # print(data[7].replace(' ',''))
            degrees = eval(data[10])
            if m in ["CEN", "GCN_2017"] and dnum == 150:
                degrees = degrees[0:(200-dnum)*20]

            dcount = []

            for d in drange:
                # print(np.size(degrees))
                dcount.append(np.sum(np.array(degrees)<=d)/np.size(degrees))

            plt.plot(drange, dcount, label=method_labels[i], linewidth=2)

        plt.grid(axis='y')
        # plt.grid(linestyle='--')
        plt.xlabel(f'Node degree $d$', fontdict={'family':'serif', 'size':14})
        plt.ylabel(f'Cumulative Degree Distribution $P_d$', fontdict={'family':'serif', 'size':14})
        plt.ylim(0, 1.03)
        plt.legend(loc='lower right')
        n = 'fig10c' if dnum==100 else 'fig10d'
        plt.savefig(f'./Figs/{n}.png', dpi=600, bbox_inches='tight')
        plt.show()


# draw Fig.11b
def draw_method_case():

    # num_destroyed = [25, 50, 75, 100, 125, 150, 175]
    num_destroyed = 100

    num_methods = []
    time = [i/10 for i in range(-15, 500)]

    linestyles = ['-', '-', '-.', '--', ':', '-.', '--', ':']

    methods = ["MDSG-GC", "MDSG-APF", "CEN", "HERO", "SIDR", "GCN_2017", "CR-MGC", "DEMD"]
    method_labels = ["MDSG-GC", "MDSG-APF", "centering", "HERO", "SIDR", "GCN", "CR-MGC", "DEMD"]

    for m in methods:
        with open(f'./Logs/case/{m}.txt', 'r') as f:
            data = f.read().split('\n')

        # print(data[7].replace(' ',''))
        num_subnet = data[0].replace(' ','').strip('[').strip(']')
        num_subnet = [int(s) for s in num_subnet.split(',')]
        # print(num_subnet)

        num_subnet = [1 for _ in range(15)] + num_subnet

        while(len(num_subnet)<len(time)):
            num_subnet.append(1)

        num_methods.append(num_subnet)

    plt.figure()

    for i, s in enumerate(num_methods):
        plt.plot(time, s, c=mcolors.TABLEAU_COLORS[colors[i]], label=method_labels[i], linestyle=linestyles[i], linewidth=2)

    plt.xlim(-1,31)
    plt.ylim(0,9)
    plt.grid(linestyle='--')
    plt.xlabel('Time $t$ /s', fontdict={'family':'serif', 'size':14})
    plt.ylabel('Number of Sub-nets', fontdict={'family':'serif', 'size':14})
    plt.legend(loc='upper right', ncol=2)
    # plt.plot(num_destroyed, step_methods[2], c=mcolors.TABLEAU_COLORS[colors[2]],marker=marklist[2])
    plt.plot(time, num_methods[1], c=mcolors.TABLEAU_COLORS[colors[1]])
    plt.plot(time, num_methods[0], c=mcolors.TABLEAU_COLORS[colors[0]])

    plt.savefig('./Figs/fig11b.png', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Fig.7
    # draw_khop()

    # # Fig.8
    # draw_batch()

    # # Fig.9
    # draw_loss_curve()

    # Fig.10
    # draw_method_figs()
    # draw_spatial_coverage()
    # draw_degree_distribution()

    # Fig.11
    draw_method_case()