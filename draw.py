import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

def sim_reward():
    with open("sim_reward.txt") as r_log:
        r = r_log.readlines()
        # r = r[0:3000]
        new_r = []
        sim_count = []
        for reward in r[:]:
            if isinstance(eval(reward), list):
                new_r.append(eval(reward)[0])
                sim_count.append(eval(reward)[1])

        plt.plot(sim_count,new_r)
        plt.show()


def draw_fill_graph():
    origin_reward = []
    origin_x = []
    step_range = 70
    for i in range(10):
        with open("sim/sim_reward_"+str(i)+".txt") as r_log:
            r = r_log.readlines()
            new_r = []
            x_axis = []
            for reward in r[:]:
                if isinstance(eval(reward), list):
                    new_r.append(eval(reward)[0])
                    x_axis.append(eval(reward)[1])
            new_r = new_r[:step_range]
            x_axis = x_axis[:step_range]
            origin_reward.append(np.array(new_r))
            origin_x.append(np.array(x_axis))
            # plt.plot(range(len(new_r)),new_r)
            # plt.show()
    origin_reward = np.array(origin_reward)
    origin_x = np.array(origin_x)
    origin_reward = np.transpose(origin_reward)
    origin_x = np.transpose(origin_x)
    print(np.max(origin_reward[-1]))
    mean_reward = np.mean(origin_reward, axis=1)
    # print mean_reward[-1]
    origin_x = np.mean(origin_x, axis=1)
    reward_std = np.std(origin_reward, axis=1)
    with open('sim/sim_reward.txt', 'w') as mean_log:
        for i in range(mean_reward.shape[0]):
            mean_log.write(str([mean_reward[i], origin_x[i]])+'\n')
    plt.plot(origin_x, mean_reward)
    # plt.plot(range(31), mean_reward+reward_std)
    # plt.plot(range(31), mean_reward-reward_std)
    plt.fill_between(origin_x, mean_reward+reward_std, mean_reward-reward_std, alpha=0.1)
    plt.xlabel('Sim Count')
    plt.ylabel('Rewards')
    plt.show()


def smooth_sim_reward(dir):
    best_assign = open(dir+'/best_assign.txt', 'r')
    sim_reward = open(dir + '/sim_reward.txt', 'r')
    sim_count = sim_reward.readlines()
    x = []
    for reward in sim_count:
        if isinstance(eval(reward), list):
            # new_r.append(eval(reward)[0])
            x.append(eval(reward)[1])

    y = []
    lines = best_assign.readlines()
    new_lines = []
    smooth = {}
    for i in range(0, len(lines), 2):
        new_lines.append([eval(lines[i]), lines[i+1]])
    for line in new_lines:
        if line[1] not in smooth:
            smooth[line[1]] = [line[0], 1]
        else:
            smooth[line[1]] = [line[0], 1+smooth[line[1]][1]]

    current_top = min(smooth.values())[0]
    # print current_top
    for line in new_lines:
        if smooth[line[1]][1]>1:
            y.append(smooth[line[1]][0])
            if smooth[line[1]][0] > current_top:
                current_top = smooth[line[1]][0]
        else:
            y.append(current_top)
    for i in range(1,len(y)):
        if y[i]<y[i-1]:
            y[i]=y[i-1]

    new_r = []
    for reward in sim_count:
        if isinstance(eval(reward), list):
            new_r.append(eval(reward)[0])

    fig = plt.figure()
    plt.plot(x,y)
    plt.plot(x,new_r)
    plt.show()
    fig.savefig('pics/'+dir+'.pdf', dpi=150, bbox_inches='tight')

def getxy(file):
    with open(file) as r_log:
        r = r_log.readlines()
        # r = r[0:3000]
        new_r = []
        sim_count = []
        for reward in r[:]:
            if isinstance(eval(reward), list):
                new_r.append(eval(reward)[0])
                sim_count.append(eval(reward)[1])

        # return sim_count, new_r

        conv_r = []
        conv_r.append((new_r[0]+new_r[1])/2)
        for i in range(1,len(new_r)-1):
            conv_r.append((new_r[i-1]+new_r[i]+new_r[i+1])/3)
        conv_r.append((new_r[-2] + new_r[-1]) / 2)

        return sim_count, conv_r
        # extra_conv = []
        # extra_conv.append(conv_r[0])
        # extra_conv.append((conv_r[0] + conv_r[1]) / 2)
        # extra_conv.append((conv_r[0] + conv_r[1] + conv_r[2]) / 3)
        # for i in range(2,len(conv_r)-1):
        #     extra_conv.append((conv_r[i-2]+conv_r[i-1]+conv_r[i]+conv_r[i+1])/4)

        alpha = 0.82
        extra_conv =[]
        extra_conv.append(conv_r[0])
        for i in range(1,len(conv_r)):
            extra_conv.append(alpha*extra_conv[i-1]+(1-alpha)*conv_r[i])

        return sim_count, extra_conv

def list_pic(file_list, split):
    fig = plt.figure()
    labels = [1,2,3,4,5,6]
    count = 0
    for file in file_list:
        x,y = getxy(file+'/sim_reward.txt')
        split_point = 0
        for i in range(len(x)):
            if x[i]>split:
                break
            split_point+=1
        labels[count], =plt.plot(x[:split_point],y[:split_point], label=file)
        # print y[split_point-1]
        count+=1
    labels[count], = plt.plot([0,split], [1891,1891], label='Human expert', linestyle=':')

    plt.legend(labels[:count+1], ['TLEA_0.75','TLEA_0.5','TLEA_0.25','Simulation','Human expert'], loc='lower right')
    plt.xlabel('Simulation Count', fontsize=18)
    plt.ylabel('Reward', fontsize=18)
    plt.xticks(range(0,split+3000,3000),fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    if 'mid' in file_list[0]:
        fig.savefig('pics/conv/32.pdf', dpi=150, bbox_inches='tight')
    else:
        fig.savefig('pics/conv/20.pdf', dpi=150, bbox_inches='tight')


def conv_smooth(dir):
    with open(dir+"/sim_reward.txt") as r_log:
        r = r_log.readlines()
        # r = r[0:3000]
        new_r = []
        sim_count = []
        for reward in r[:]:
            if isinstance(eval(reward), list):
                new_r.append(eval(reward)[0])
                sim_count.append(eval(reward)[1])
        conv_r = []
        conv_r.append((new_r[0]+new_r[1])/2)
        for i in range(1,len(new_r)-1):
            conv_r.append((new_r[i-1]+new_r[i]+new_r[i+1])/3)
        conv_r.append((new_r[-2] + new_r[-1]) / 2)

        fig=plt.figure()
        plt.plot(sim_count,conv_r, c='b')

        with open(dir + "/ingoing_noble.txt") as I_log:
            I = I_log.readlines()
            I = I[len(I)-len(conv_r):]
            new_I = []
            for reward in I:
                new_I.append(eval(reward))
            conv_I = []
            conv_I.append((new_I[0] + new_I[1]) / 2)
            for i in range(1, len(new_I) - 1):
                conv_I.append((new_I[i - 1] + new_I[i] + new_I[i + 1]) / 3)
            conv_I.append((new_I[-2] + new_I[-1]) / 2)

            plt.plot(sim_count, new_I, c='g')
            # plt.show()
        # plt.show()
        fig.savefig('pics/conv/' + dir + '_conv.pdf', dpi=150, bbox_inches='tight')


def blood():
    nb_blood = 5
    b = np.zeros((nb_blood, 100),dtype=np.float32)
    count = 0
    with open('blood_log2.txt') as b_log:
        for line in b_log.readlines():
            line = line.split()
            for i in range(nb_blood):
                b[i][count] = eval(line[i])
            count+=1

    fig = plt.figure()
    labels = ['noble 0','civil 0','civil 1000','civil 2000','civil 3000']
    labels = [i for i in range(nb_blood)]
    for i in range(nb_blood):
        labels[i] = str(labels[i])
    for i in range(nb_blood):
        plt.plot(range(100), b[i], alpha=0.8, label=labels[i])
    plt.legend()
    plt.show()

# dir_list = ['noble_0.75_mid','noble_0.5_mid','noble_0.25_mid','noble_0.75','noble_0.5','noble_0.25','noble_0.75_1000',
#             'sim_mid', 'sim']
# sim_reward()
# smooth_sim_reward('noble_0.75')
# for dir_name in dir_list:
#     conv_smooth(dir_name)
# conv_smooth('noble_0.5')
# list_pic(['noble_0.75','noble_0.5','noble_0.25', 'sim'], 17500)
# list_pic(['noble_0.75_mid','noble_0.5_mid','noble_0.25_mid','sim_mid'],20000)
# draw_fill_graph()

blood()