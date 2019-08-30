import matplotlib

# matplotlib.use('Agg')
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
import time
import os
import imageio
import config


class ResultVisualizer:
    def __init__(self,mapsize, source_pos, hole_pos, hole_city, city_dis, agent_num, directory, trans):
        self.mapsize = mapsize
        self.source_pos = source_pos
        self.hole_pos = hole_pos
        self.hole_city = hole_city
        self.city_dis = city_dis
        self.directory = directory
        self.enable = False
        self.trans = trans
        if agent_num!=-1:
            self.remove_files()

    def remove_files(self):
        if os.path.exists(self.directory):
            if os.path.exists(self.directory+ '/observation'):
                for file in os.listdir(self.directory + '/observation'):
                    os.remove(self.directory + '/observation/' + file)
                # os.rmdir(self.directory + '/observation')
            else:
                os.mkdir(self.directory + '/observation')
            for file in os.listdir(self.directory):
                try:
                    os.remove(self.directory + '/' +file)
                except:
                    pass
            # os.rmdir(self.directory)
        # os.mkdir(self.directory)
        # os.mkdir(self.directory + '/observation')

    def wirte_static_info(self):
        static_info = open(self.directory + '/static_info', 'w')
        static_info.write(str(self.mapsize) + '\n')
        static_info.write(str(self.city_dis) + '\n')
        static_info.write(str(self.source_pos) + '\n')
        static_info.write(str(self.hole_pos) + '\n')
        if isinstance(self.hole_city, list):
            static_info.write(str(self.hole_city) + '\n')
        else:
            static_info.write(str(self.hole_city.tolist()) + '\n')
        if isinstance(self.trans, list):
            static_info.write(str(self.trans) + '\n')
        else:
            static_info.write(str(self.trans.tolist()) + '\n')
        static_info.close()


    def write_ob(self, step, agent_pos, agent_city, agent_reward, hole_reward, source_reward, reward):
        episode = (step-1) / config.Game.total_time
        log = open(self.directory + '/observation/'+str(episode), 'a')
        log.write(str(agent_reward) + '\n')
        log.write(str(source_reward) + '\n')
        log.write(str(hole_reward) + '\n')
        log.write(str(agent_pos) + '\n')
        log.write(str(agent_city) + '\n')
        log.write(str(reward) + '\n')
        log.close()

    def write_reward(self, epi_reward):
        log = open(self.directory + '/rewards','a')
        log.write(str(epi_reward.tolist()) + '\n')
        log.close()

    def randomcolor(self, agent_num):
        colors = np.random.random((agent_num+1, 3))
        return colors

    def draw_map(self, mapsize, conveyors, hole_pos, hole_city, agent_pos, agent_city, colors,
                 dir, filename, step, agent_reward, hole_reward, source_reward, city_dis, reward, trans):
        fig = plt.figure()

        # fontsize for texts
        fontsize = int(120.0/mapsize[0])

        # main graph
        ax = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=3)
        ax.grid(True, color="black", alpha=0.25, linestyle='solid')

        ax1 = plt.subplot2grid((3, 4), (0, 3))
        ax1.text(0, 0.8, "map size: " + str(mapsize), size=12, weight="light")
        ax1.text(0, 0.6, "hole num: " + str(len(hole_pos)), size=12, weight="light")
        ax1.text(0, 0.4, "source num: " + str(len(conveyors)), size=12, weight="light")
        ax1.text(0, 0.2, "agent num: " + str(len(agent_pos)), size=12, weight="light")
        ax1.text(0, 0, "city distribution: ", size=12, weight="light")

        # pie graph
        ax2 = plt.subplot2grid((3, 4), (1, 3))
        ax2.pie(city_dis, colors=colors, radius=1.25, autopct='%1.1f%%')

        # timestep graph
        ax3 = plt.subplot2grid((3, 4), (2, 3))
        ax3.text(0, 0.8, "Timestep: " + str(step / 2), size=12, weight="light")
        ax3.text(0, 0.6, "Pack num: " + str(sum(source_reward)), size=12, weight="light")
        ax3.text(0, 0.4, "Reward: " + str(float(int(reward * 100)) / 100), size=12, weight="light")
        if step / 2 == 0:
            ax3.text(0, 0.2, "R/T: " + str(float(int(reward * 100)) / 100), size=12, weight="light")
        else:
            ax3.text(0, 0.2, "R/T: " + str(float(int(reward * 1.0 / (step / 2) * 100)) / 100), size=12, weight="light")

        for k in range(len(conveyors)):
            p=patches.Polygon(
                [[conveyors[k][0],conveyors[k][1]],
                 [conveyors[k][0]+1, conveyors[k][1]],
                 [conveyors[k][0]+0.5, conveyors[k][1]+1]
                 ],
                facecolor=(0.9,0.9,0.9),
                linewidth=0.5,
                linestyle='solid'
            )
            ax.text(conveyors[k][0]+0.35, conveyors[k][1]+0.35, str(source_reward[k]),
                    size=fontsize, weight="light")
            ax.add_patch(p)

        for i in range(len(hole_pos)):
            p = patches.Rectangle(
                ((hole_pos[i][0]), (hole_pos[i][1])),
                1,
                1,
                facecolor=(colors[hole_city[i]][0], colors[hole_city[i]][1], colors[hole_city[i]][2]),
                linewidth=0.5,
                linestyle='solid'
            )
            ax.text(hole_pos[i][0]+0.35, hole_pos[i][1]+0.35, str(hole_reward[i]),
                    size=fontsize, weight="light", color=(1,1,1))
            ax.add_patch(p)

        for j in range(len(agent_pos)):
            p = patches.Circle(
                ((agent_pos[j][0]+0.5), (agent_pos[j][1]+0.5)),
                0.4,
                facecolor=(colors[agent_city[j]][0], colors[agent_city[j]][1], colors[agent_city[j]][2]),
                linewidth=0.5,
                linestyle='solid'
            )
            ax.text(agent_pos[j][0]+0.35, agent_pos[j][1]+0.35, str(agent_reward[j]),
                    size=fontsize, weight="light", alpha=0.85)
            ax.add_patch(p)

        for i in range(mapsize[0]):
            for j in range(mapsize[1]):
                if trans[i][j][0]==1:
                    p = patches.Arrow(i + 0.8, j + 0.5, 0.15, 0., 0.15, facecolor=(0.3, 0.3, 0.3),
                        linewidth=0.01, linestyle='solid')
                    ax.add_patch(p)
                if trans[i][j][1]==1:
                    p = patches.Arrow(i + 0.5, j + 0.8, 0., 0.15, 0.15, facecolor=(0.3, 0.3, 0.3),
                        linewidth=0.01, linestyle='solid')
                    ax.add_patch(p)
                if trans[i][j][2]==1:
                    p = patches.Arrow(i + 0.2, j + 0.5, -0.15, 0., 0.15, facecolor=(0.3, 0.3, 0.3),
                        linewidth=0.01, linestyle='solid')
                    ax.add_patch(p)
                if trans[i][j][3]==1:
                    p = patches.Arrow(i + 0.5, j + 0.2, 0., -0.15, 0.15, facecolor=(0.3, 0.3, 0.3),
                        linewidth=0.01, linestyle='solid')
                    ax.add_patch(p)

        # set ticks and spines
        ax.set_xticks(np.arange(0, mapsize[0]+1, 1))
        ax.set_xticklabels(())
        ax.set_yticks(np.arange(0, mapsize[1]+1, 1))
        ax.set_yticklabels(())

        ax1.spines['bottom'].set_color('none')
        ax1.spines['top'].set_color('none')
        ax1.spines['right'].set_color('none')
        ax1.spines['left'].set_color('none')
        ax1.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',
                        right='off', left='off', labelleft='off')

        ax2.spines['bottom'].set_color('none')
        ax2.spines['top'].set_color('none')
        ax2.spines['right'].set_color('none')
        ax2.spines['left'].set_color('none')
        ax2.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',
                        right='off', left='off', labelleft='off')

        ax3.spines['bottom'].set_color('none')
        ax3.spines['top'].set_color('none')
        ax3.spines['right'].set_color('none')
        ax3.spines['left'].set_color('none')
        ax3.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',
                        right='off', left='off', labelleft='off')

        # plt.show()
        if not os.path.exists(dir):
            os.mkdir(dir)
        fig.savefig(dir + '/' + filename + str(step) + '.png', dpi=100, bbox_inches='tight')
        plt.close(fig)

    def save_gif(self, dir, filename, step):
        with imageio.get_writer(dir+'/'+filename+'.gif',mode='I',fps=10) as writer:
            for i in range(step):
                image = imageio.imread(dir+'/'+filename+str(i)+'.png')
                writer.append_data(image)

    def save_mp4(self, dir, filename, step):
        with imageio.get_writer(dir+'/'+filename+'.mp4',mode='I',fps=10) as writer:
            for i in range(step):
                image = imageio.imread(dir+'/'+filename+str(i)+'.png')
                writer.append_data(image)

    def draw_log(self, pic_nb=69):
        # self.draw_reward()

        log = open(self.directory + '/static_info', 'r')
        mapsize = eval(log.readline())
        city_dis = eval(log.readline())
        source_pos = eval(log.readline())
        hole_pos = eval(log.readline())
        hole_city = eval(log.readline())
        trans = eval(log.readline())
        colors = self.randomcolor(len(city_dis))
        colors[len(city_dis)] = [0.9, 0.9, 0.9]
        log.close()

        for file in os.listdir(self.directory + '/observation'):
            log = open(self.directory + '/observation/' + file)

            step = 0
            agent_reward = eval(log.readline())
            source_reward = eval(log.readline())
            hole_reward = eval(log.readline())
            agent_pos = eval(log.readline())
            agent_city = eval(log.readline())
            reward = eval(log.readline())
            self.draw_map(mapsize, source_pos, hole_pos, hole_city, agent_pos, agent_city, colors,
                     self.directory+"/pics/"+file, "demo", step, agent_reward, hole_reward, source_reward,
                     city_dis, reward, trans)

            # old_agent_reward = agent_reward
            # old_source_reward = source_reward
            # old_hole_reward = hole_reward
            old_agent_pos = agent_pos
            # old_agent_city = agent_city

            for j in range(pic_nb):
                step += 1
                agent_reward = eval(log.readline())
                source_reward = eval(log.readline())
                hole_reward = eval(log.readline())
                agent_pos = eval(log.readline())
                agent_city = eval(log.readline())
                reward += eval(log.readline())
                for i in range(len(agent_pos)):
                    old_agent_pos[i][0] = (agent_pos[i][0] + old_agent_pos[i][0]) / 2.0
                    old_agent_pos[i][1] = (agent_pos[i][1] + old_agent_pos[i][1]) / 2.0
                self.draw_map(mapsize, source_pos, hole_pos, hole_city, old_agent_pos, agent_city, colors,
                              self.directory+"/pics/"+file, "demo", step, agent_reward, hole_reward, source_reward,
                              city_dis, reward, trans)
                step += 1
                old_agent_pos = agent_pos
                self.draw_map(mapsize, source_pos, hole_pos, hole_city, agent_pos, agent_city, colors,
                              self.directory+"/pics/"+file, "demo", step, agent_reward, hole_reward, source_reward,
                              city_dis, reward, trans)

            log.close()
            try:
                self.save_mp4(self.directory+"/pics/"+file, "demo", 2 * pic_nb + 1)
            except:
                pass

    def draw_reward(self):
        log = open(self.directory + '/rewards','r')
        rewards = []
        r = log.readline()
        while r != '':
            r = eval(r)
            if type(r)==type([]):
                rewards.append(sum(r))
            else:
                rewards.append(r)
            r = log.readline()
        log.close()

        fig = plt.figure()
        plt.plot(range(len(rewards)),rewards)
        fig.savefig(self.directory + '/rewards.png', dpi=100, bbox_inches='tight')
        plt.close(fig)

def draw_env_reward():
    log = open('../reward_his', 'r')

    rewards = []
    r = log.readline()
    while r != '':
        r = eval(r)
        if type(r) == type([]):
            rewards.append(sum(r))
        else:
            rewards.append(r)
        r = log.readline()
    log.close()

    fig = plt.figure()
    plt.plot(range(len(rewards)), rewards)
    fig.savefig('../rewards.png', dpi=100, bbox_inches='tight')
    plt.close(fig)

def draw_thermal(w, therm, name, reward):
    log = open('result/static_info', 'r')
    mapsize = eval(log.readline())
    city_dis = eval(log.readline())
    source_pos = eval(log.readline())
    hole_pos = eval(log.readline())
    hole_city = eval(log.readline())
    trans = eval(log.readline())
    colors = np.random.random((len(city_dis) + 1, 3))
    colors[len(city_dis)] = [0.9, 0.9, 0.9]
    log.close()

    fig = plt.figure()

    # fontsize for texts
    fontsize = int(120.0 / mapsize[0])

    # main graph
    ax = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=3)
    ax.grid(True, color="black", alpha=0.25, linestyle='solid')

    ax1 = plt.subplot2grid((3, 4), (0, 3))
    ax1.text(0, 0.8, "map size: " + str(mapsize), size=12, weight="light")
    ax1.text(0, 0.6, "hole num: " + str(len(hole_pos)), size=12, weight="light")
    ax1.text(0, 0.4, "source num: " + str(len(source_pos)), size=12, weight="light")
    ax1.text(0, 0.2, "agent num: " + str(2*20+20), size=12, weight="light")
    ax1.text(0, 0, "reward: "+str(reward), size=12, weight="light")
    ax1.text(0, -0.2, "city distribution: ", size=12, weight="light")

    # pie graph
    ax2 = plt.subplot2grid((3, 4), (1, 3))
    ax2.pie(city_dis, colors=colors, radius=1.25, autopct='%1.1f%%')

    therm = therm/np.max(therm)

    for i in range(mapsize[0]):
        for j in range(mapsize[1]):
            q = patches.Rectangle((i, j), 1, 1, facecolor=(1, 1 - therm[i][j], 1 - therm[i][j]),alpha=1)
            ax.add_patch(q)

    for k in range(len(source_pos)):
        p = patches.Polygon(
            [[source_pos[k][0], source_pos[k][1]],
             [source_pos[k][0] + 1, source_pos[k][1]],
             [source_pos[k][0] + 0.5, source_pos[k][1] + 1]
             ],
            facecolor=(0.9, 0.9, 0.9),
            linewidth=0.5,
            linestyle='solid'
        )
        ax.add_patch(p)

    for i in range(len(hole_pos)):
        p = patches.Rectangle(
            ((hole_pos[i][0]+0.2), (hole_pos[i][1]+0.2)),
            0.6,
            0.6,
            facecolor=(colors[hole_city[i]][0], colors[hole_city[i]][1], colors[hole_city[i]][2]),
            linewidth=0.5,
            linestyle='solid'
        )
        ax.add_patch(p)

    for i in range(mapsize[0]):
        for j in range(mapsize[1]):
            if trans[i][j][0] == 1:
                p = patches.Arrow(i + 0.8, j + 0.5, 0.15, 0., 0.15, facecolor=(0.3, 0.3, 0.3),
                                  linewidth=0.01, linestyle='solid')
                ax.add_patch(p)

            if trans[i][j][1] == 1:
                p = patches.Arrow(i + 0.5, j + 0.8, 0., 0.15, 0.15, facecolor=(0.3, 0.3, 0.3),
                                  linewidth=0.01, linestyle='solid')
                ax.add_patch(p)
            if trans[i][j][2] == 1:
                p = patches.Arrow(i + 0.2, j + 0.5, -0.15, 0., 0.15, facecolor=(0.3, 0.3, 0.3),
                                  linewidth=0.01, linestyle='solid')
                ax.add_patch(p)
            if trans[i][j][3] == 1:
                p = patches.Arrow(i + 0.5, j + 0.2, 0., -0.15, 0.15, facecolor=(0.3, 0.3, 0.3),
                                  linewidth=0.01, linestyle='solid')
                ax.add_patch(p)

    # set ticks and spines
    ax.set_xticks(np.arange(0, mapsize[0] + 1, 1))
    ax.set_xticklabels(())
    ax.set_yticks(np.arange(0, mapsize[1] + 1, 1))
    ax.set_yticklabels(())

    ax1.spines['bottom'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.spines['right'].set_color('none')
    ax1.spines['left'].set_color('none')
    ax1.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',
                    right='off', left='off', labelleft='off')

    ax2.spines['bottom'].set_color('none')
    ax2.spines['top'].set_color('none')
    ax2.spines['right'].set_color('none')
    ax2.spines['left'].set_color('none')
    ax2.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',
                    right='off', left='off', labelleft='off')
    # plt.show()

    fig.savefig('result/therm_pic/' + name + '.png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    plt.cla()

if __name__ == '__main__':
    # log = open('result/static_info', 'r')
    # mapsize = eval(log.readline())
    # city_dis = eval(log.readline())
    # source_pos = eval(log.readline())
    # hole_pos = eval(log.readline())
    # hole_city = eval(log.readline())
    # visualizer = ResultVisualizer(mapsize, source_pos, hole_pos,
    #                               hole_city, city_dis, -1, "result")
    # log.close()
    # visualizer.draw_log()

    draw_env_reward()