import gym
import numpy as np
# from agent_gym import AGENT_GYM
import config, utils
# import matplotTest as draw
# import mapGenerator as MG
import copy
import Queue
import random
from scipy.signal import convolve2d
import os

trans_to_action = [[1,0],[0,1],[-1,0],[0,-1]]

class GreedyAstar:
    def __init__(self, source_pos, hole_pos, hole_city, agent_num, trans=None):
        self.source_pos = source_pos
        self.hole_pos = hole_pos
        self.hole_city = hole_city
        self.agent_num = agent_num
        self.agent_pos = None
        self.trans = trans
        self.source_hole_map = self.get_map()
        # self.distance = self.get_all_distance()
        if config.directions is None:
            if os.path.exists('directions' + str(config.Map.Width) + '.txt'):
                print "start directions"
                direc = open('directions' + str(config.Map.Width) + '.txt', 'r')
                self.directions = np.array(eval(direc.readline()))
                config.directions = copy.deepcopy(self.directions)
                print "finish directions"
            else:
                print "start directions"
                self.directions = self.getDirections()
                direc = open('directions' + str(config.Map.Width) + '.txt', 'w')
                direc.write(str(self.directions.tolist()))
                config.directions = copy.deepcopy(self.directions)
                print "finish directions"
        else:
            self.directions = config.directions
        self.agent_city = [-1 for _ in range(agent_num)]
        self.start_pos = [-1 for _ in range(agent_num)]
        self.end_pos = [-1 for _ in range(agent_num)]
        self.end_update = [True for _ in range(agent_num)]

        self.dis_sum = np.zeros((config.Source_num, config.Hole_num, 2))
        self.dis_count = np.ones((config.Source_num, config.Hole_num, 2))
        self.dis_average = np.zeros((config.Source_num, config.Hole_num, 2))
        self.step_count = np.zeros((self.agent_num,))
        self.init_distance()

    def init_distance(self):
        for i in range(config.Source_num):
            for j in range(config.Hole_num):
                dis = abs(self.source_pos[i][0]-self.hole_pos[j][0]) + abs(self.source_pos[i][1]-self.hole_pos[j][1])
                self.dis_sum[i][j] = 1.2 * np.array([dis,dis])
                self.dis_average[i][j] = 1.2 * np.array([dis,dis])

    def get_map(self):
        map = -2*np.ones((config.Map.Width, config.Map.Height))
        for source in self.source_pos:
            map[source[0]][source[1]] = -1
        for i in range(len(self.hole_pos)):
            map[self.hole_pos[i][0]][self.hole_pos[i][1]] = self.hole_city[i]
        return map


    def getDirections(self):
        directions = -np.ones((config.Source_num+config.Hole_num, config.Map.Width, config.Map.Height),dtype=np.int32)
        queue = Queue.Queue(maxsize=config.Map.Width * config.Map.Height)
        for i in range(config.Source_num):
            queue.put(self.source_pos[i])
            while not queue.empty():
                current = queue.get()
                for j in range(4):
                    if self.trans[current[0]][current[1]][j] == 0:
                        next_pos = [current[0]+trans_to_action[j][0], current[1]+trans_to_action[j][1]]
                        if utils.inMap(next_pos) and directions[i][next_pos[0]][next_pos[1]] == -1:
                            directions[i][next_pos[0]][next_pos[1]] = (j+2)%4
                            if self.source_hole_map[next_pos[0]][next_pos[1]] == -2:
                                queue.put(next_pos)

        for i in range(config.Source_num,config.Source_num+config.Hole_num):
            queue.put(self.hole_pos[i-config.Source_num])
            while not queue.empty():
                current = queue.get()
                for j in range(4):
                    if self.trans[current[0]][current[1]][j] == 0:
                        next_pos = [current[0]+trans_to_action[j][0], current[1]+trans_to_action[j][1]]
                        if utils.inMap(next_pos) and directions[i][next_pos[0]][next_pos[1]] == -1:
                            directions[i][next_pos[0]][next_pos[1]] = (j+2)%4
                            if self.source_hole_map[next_pos[0]][next_pos[1]] == -2:
                                queue.put(next_pos)
        return directions

    def getJointAction(self, agent_pos, agent_city):
        self.agent_pos = agent_pos
        self.agent_city = agent_city

        self.updateEnd()


        next_agent_pos = []
        for i in range(self.agent_num):
            action = trans_to_action[self.directions[self.end_pos[i]][self.agent_pos[i][0]][self.agent_pos[i][1]]]
            next_agent_pos.append([self.agent_pos[i][0] + action[0],
                                   self.agent_pos[i][1] + action[1]])

        # print self.agent_pos
        # print next_agent_pos
        # print "old action:", action
        action = self.check_path(next_agent_pos)

        # print "action:", action
        return action


    def updateEnd(self):
        self.dense =np.zeros((config.Map.Width, config.Map.Height),dtype=np.int32)
        for i in range(self.agent_num):
            self.dense[self.agent_pos[i][0]][self.agent_pos[i][1]] += 1
        self.dense = convolve2d(self.dense, np.ones((5,5)), mode='same')

        self.step_count += np.ones((self.agent_num,))
        for i in range(self.agent_num):
            if self.end_update[i]:
                self.end_update[i] = False
                if self.start_pos[i]==-1:
                    to_hole = 0
                else:
                    to_hole = int(self.start_pos[i] < self.end_pos[i])
                # print i, self.start_pos[i], self.end_pos[i], to_hole
                # if self.start_pos[i]!=-1:
                #     if to_hole:
                #         self.dis_count[self.start_pos[i]][self.end_pos[i]-config.Source_num][to_hole] += 1
                #         self.dis_sum[self.start_pos[i]][self.end_pos[i]-config.Source_num][to_hole] += self.step_count[i]
                #         self.dis_average[self.start_pos[i]][self.end_pos[i]-config.Source_num][to_hole] = \
                #             self.dis_sum[self.start_pos[i]][self.end_pos[i]-config.Source_num][to_hole] / \
                #             self.dis_count[self.start_pos[i]][self.end_pos[i]-config.Source_num][to_hole]
                #     else:
                #         self.dis_count[self.end_pos[i]][self.start_pos[i]-config.Source_num][to_hole] += 1
                #         self.dis_sum[self.end_pos[i]][self.start_pos[i]-config.Source_num][to_hole] += self.step_count[i]
                #         self.dis_average[self.end_pos[i]][self.start_pos[i]-config.Source_num][to_hole] = \
                #             self.dis_sum[self.end_pos[i]][self.start_pos[i]-config.Source_num][to_hole] / \
                #             self.dis_count[self.end_pos[i]][self.start_pos[i]-config.Source_num][to_hole]
                self.step_count[i] = 0
                self.start_pos[i] = self.end_pos[i]
                min_dis = 9999999
                min_index = -1
                if self.start_pos[i] == -1:
                    for j in range(config.Source_num):
                        if abs(self.source_pos[j][0]-self.agent_pos[i][0]) + abs(self.source_pos[j][1]-self.agent_pos[i][1]) < min_dis:
                            min_dis = abs(self.source_pos[j][0]-self.agent_pos[i][0]) + abs(self.source_pos[j][1]-self.agent_pos[i][1])
                            min_index = j
                    self.end_pos[i] = min_index
                    continue
                if to_hole:
                    # print "find source", i
                    for j in range(config.Source_num):
                        dense = self.dense[self.source_pos[j][0]][self.source_pos[j][1]]
                        if self.dis_average[j][self.start_pos[i] - config.Source_num][0] + dense < min_dis:
                            min_dis = self.dis_average[j][self.start_pos[i]- config.Source_num][0] + dense
                            min_index = j
                    self.end_pos[i] = min_index
                    # p = np.zeros((config.Source_num,))
                    # for j in range(config.Source_num):
                    #     p[j] = self.dis_average[j][self.start_pos[i] - config.Source_num][0]
                    # p = softmax(p)
                    # self.end_pos[i] = np.random.choice(range(config.Source_num), p=p)
                else:
                    # print "find hole", i
                    # print self.start_pos[i]
                    for j in range(config.Hole_num):
                        dense = self.dense[self.hole_pos[j][0]][self.hole_pos[j][1]]
                        if self.hole_city[j] == self.agent_city[i] and self.dis_average[self.start_pos[i]][j][1] + dense< min_dis:
                            min_dis = self.dis_average[self.start_pos[i]][j][1] + dense
                            min_index = j
                    self.end_pos[i] = min_index + config.Source_num
        # print self.end_pos

    def fun(self, done, i, new_pos, agents, agent_map):
        done[i] = True
        pre_pos = self.agent_pos[i]
        agents[i] = new_pos[i]
        agent_id = agent_map[agents[i][0]][agents[i][1]]
        if agent_id>=0:
            if done[agent_id]:
                agents[i] = pre_pos
                return False
            else:
                agent_map[pre_pos[0]][pre_pos[1]] = -1
                agent_map[agents[i][0]][agents[i][1]] = i
                if self.fun(done, agent_id, new_pos, agents, agent_map):
                    return True
                else:
                    agents[i] = pre_pos
                    agent_map[pre_pos[0]][pre_pos[1]] = i
                    agent_map[agents[i][0]][agents[i][1]] = agent_id
                    return False
        elif (self.source_hole_map[agents[i][0]][agents[i][1]] != -2 and
                            self.source_hole_map[agents[i][0]][agents[i][1]]!=self.agent_city[i]):
            agents[i] = pre_pos
            return False
        elif agent_map[agents[i][0]][agents[i][1]] == -1:
            agent_map[agents[i][0]][agents[i][1]] = i
            return True

    def check_path(self, new_pos):
        done = [False for i in range(self.agent_num)]
        agent_map=-np.ones((config.Map.Width, config.Map.Height),dtype=np.int32)
        agents_pos = copy.deepcopy(self.agent_pos)
        for i in range(self.agent_num):
            agent_map[self.agent_pos[i][0]][self.agent_pos[i][1]] = i

        for i in range(self.agent_num):
            if not done[i]:
                self.fun(done, i, new_pos, agents_pos, agent_map)

        # print done
        action = []
        for i in range(self.agent_num):
            action.append([agents_pos[i][0]-self.agent_pos[i][0], agents_pos[i][1]-self.agent_pos[i][1]])
        return action

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    pass  # TODO: Compute and return softmax(x)
    x = np.array(x)
    x = np.exp(x)
    x.astype('float32')
    if x.ndim == 1:
        sumcol = sum(x)
        for i in range(x.size):
            x[i] = x[i] / float(sumcol)
    if x.ndim > 1:
        sumcol = x.sum(axis=0)
        for row in x:
            for i in range(row.size):
                row[i] = row[i] / float(sumcol[i])
    return x