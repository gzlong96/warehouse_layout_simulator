import numpy as np
import config, utils, copy
import time as ttime

import gym
from gym.utils import seeding
from whca import WHCA
from greedyastar import GreedyAstar

class AGENT_GYM(gym.Env):
    metadata = {'render.modes': ['human']}

    def init(self, hole_pos, source_pos, agent_num):
        self.time = 0
        self.agent_pos = []
        for i in range(agent_num):
            while True:
                x = np.random.random_integers(0, config.Map.Width - 1)
                y = np.random.random_integers(0, config.Map.Height - 1)
                if [x, y] not in hole_pos and [x, y] not in source_pos and [x, y] not in self.agent_pos:
                    self.agent_pos.append([x, y])
                    break
        self.agent_city = [-1] * agent_num
        self.end_count = 0
        self.window = 8
        # self.whca = WHCA(self.window, source_pos, hole_pos, self.hole_city, agent_num, [0, 1], self.trans)
        self.astar = GreedyAstar(source_pos, hole_pos, self.hole_city, agent_num, self.trans)
        self.agent_reward = [0] * agent_num

        self.therm = np.zeros((config.Map.Width, config.Map.Height), dtype=np.int32)

    def genCity(self, city_dis):
        return np.random.multinomial(1, city_dis, size=1).tolist()[0].index(1)

    def __init__(self, source_pos, hole_pos, agent_num, total_time, hole_city, city_dis, trans=None):

        self._seed()

        self.source_pos = source_pos
        self.total_time = total_time
        self.hole_pos = hole_pos
        self.hole_city = hole_city
        self.agent_num = agent_num
        self.city_dis = city_dis

        if trans is None:
            self.trans = np.zeros((config.Map.Width, config.Map.Height, 4))
        else:
            self.trans = trans

        self.init(self.hole_pos, self.source_pos, self.agent_num)

        self.steps = 0


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.init(self.hole_pos, self.source_pos, self.agent_num)
        return [1]

    def _step(self, action):
        dir = [[1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]]
        rewards = [0.0]*self.agent_num
        pick_drop = 1
        hit_wall = 0
        illegal = 0

        agent_next_pos = []
        done = [False] * len(action)
        # astar_action = self.whca.getJointAction(self.agent_pos, self.agent_city, [[-1,-1]]*len(action))
        astar_action = self.astar.getJointAction(self.agent_pos, self.agent_city)

        self.steps += 1
        # invalid
        for i in range(self.agent_num):
            pos = self.agent_pos[i]
            a = astar_action[i]
            # a = dir[action[i]]
            if a!=[0,0] and self.trans[self.agent_pos[i][0]][self.agent_pos[i][1]][dir.index(a)] == 0:
                print "illegal"
                rewards[i] -= illegal
            # TODO simple resolution
            next_pos = [pos[0] + a[0], pos[1] + a[1]]
            if next_pos not in agent_next_pos and next_pos not in self.agent_pos:
                agent_next_pos.append(next_pos)
            else:
                agent_next_pos.append(pos)
            if pos == agent_next_pos[i]:
                done[i] = True
            elif not utils.inMap(agent_next_pos[i]):
                agent_next_pos[i] = self.agent_pos[i]
                done[i] = True
                rewards[i] -= hit_wall
            elif agent_next_pos[i] in self.source_pos and self.agent_city[i] != -1:
                agent_next_pos[i] = self.agent_pos[i]
                done[i] = True
                rewards[i] -= hit_wall
            elif agent_next_pos[i] in self.hole_pos and self.agent_city[i] != self.hole_city[
                self.hole_pos.index(agent_next_pos[i])]:
                agent_next_pos[i] = self.agent_pos[i]
                done[i] = True
                rewards[i] -= hit_wall

        # circle
        for i in range(self.agent_num):
            if done[i]:
                continue
            circle = []
            j = i
            while not done[j] and j not in circle and agent_next_pos[j] in self.agent_pos:
                circle.append(j)
                j = self.agent_pos.index(agent_next_pos[j])
            if len(circle) > 0 and j == circle[0]:
                if len(circle) == 1:
                    print 'error: len(circle) == 1'
                if len(circle) == 2:
                    agent_next_pos[circle[0]] = self.agent_pos[circle[0]]
                    agent_next_pos[circle[1]] = self.agent_pos[circle[1]]
                    done[circle[0]] = True
                    done[circle[1]] = True
                else:
                    for k in range(len(circle)):
                        done[circle[k]] = True

        # line
        for i in range(self.agent_num):
            if done[i]:
                continue
            line = []
            j = i
            while not done[j] and agent_next_pos[j] in self.agent_pos:
                if j in line:
                    print 'error: duplicate in line'
                    print i, j
                    print line
                    print self.agent_pos
                    print agent_next_pos
                    print done
                line.append(j)
                j = self.agent_pos.index(agent_next_pos[j])
            if not done[j]:
                line.append(j)
                collision = False
                for k in range(self.agent_num):
                    if done[k] and agent_next_pos[k] == agent_next_pos[j]:
                        collision = True
                        break
                for k in range(len(line)):
                    if collision:
                        agent_next_pos[line[k]] = self.agent_pos[line[k]]
                    done[line[k]] = True

        if False in done:
            print 'error: False in done'
            print self.agent_pos
            print agent_next_pos
            print done

        for i in range(self.agent_num):
            if self.agent_pos[i] == agent_next_pos[i]:
                if np.random.uniform()<0.5:
                    tran = self.trans[self.agent_pos[i][0]][self.agent_pos[i][1]]
                    for j in range(4):
                        if tran[j] == 1:
                            direction = dir[j]
                            next_pos = [agent_next_pos[i][0]+direction[0],agent_next_pos[i][1]+direction[1]]
                            if next_pos not in agent_next_pos and utils.inMap(next_pos):
                                agent_next_pos[i] = [agent_next_pos[i][0]+direction[0],agent_next_pos[i][1]+direction[1]]
                                # self.astar.end_update[i] = True
                                break

        self.agent_pos = agent_next_pos

        pack_count = []

        for i in range(self.agent_num):
            pack_count.append(0)
            pos = self.agent_pos[i]

            self.therm[self.agent_pos[i][0]][self.agent_pos[i][1]] += 1
            # a = action[i]
            # if a == [0, 0]:
            #     continue
            if pos in self.source_pos and self.agent_city[i] == -1:  # source
                source_idx = self.source_pos.index(pos)
                self.agent_city[i] = self.genCity(self.city_dis)
                rewards[i] += pick_drop
                self.astar.end_update[i] = True
            elif pos in self.hole_pos and self.agent_city[i] != -1:  # hole
                hole_idx = self.hole_pos.index(pos)
                self.agent_city[i] = -1
                pack_count[-1] = 1
                rewards[i] += pick_drop
                self.astar.end_update[i] = True
            self.agent_reward[i] += rewards[i]

        # for r0 in rewards:
        #     if r0 > 0:
        #         self.end_count = 0
        # self.end_count += 1
        self.time += 1
        self.steps += 1
        if self.time == self.total_time:
            done = True
            # for i in range(self.therm.shape[0]):
            #     self.therm[i] = self.therm[i]/np.max(self.therm[i])
            # ther_log = open(
            #     "environment/result/thers/thermal" + str(config.Map.Width) + '_' + str(config.episode) + '_' + str(
            #         config.epi_of_epi), 'w')
            # ther_log.write(str(self.hole_city) + '\n')
            # ther_log.write(str(sum(self.agent_reward)) + '\n')
            # ther_log.write(str(self.therm.tolist()) + '\n')
            # ther_log.close()
            # config.data.append([copy.deepcopy(self.hole_city), sum(self.agent_reward), copy.deepcopy(self.therm)])
            # print [self.hole_city, sum(self.agent_reward)]
            return [1], np.array(rewards), done, [copy.deepcopy(self.hole_city), sum(self.agent_reward), copy.deepcopy(self.therm)]
        else:
            done = False


        return [1], np.array(rewards), done, {}