import random
from functools import cmp_to_key
import time
import config
from agent.agent_gym import AGENT_GYM
from agent.simple_agent import Agent
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Input, Flatten, Conv2DTranspose, Reshape, Concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K
import keras.callbacks
import copy
import multiprocessing as mp
import hot_supervise as EvalNet

# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF

# conf = tf.ConfigProto()
# conf.gpu_options.allow_growth=True
# sess = tf.Session(config=conf)
# KTF.set_session(sess)

random.seed(1234567)
length = len(config.Map.hole_pos)
city_dis = config.Map.city_dis
np_city_dis = np.array(city_dis)

config.Type_num = len(city_dis)+1
config.Source_num = len(config.Map.source_pos)
config.Hole_num = len(config.Map.hole_pos)

# dis = open("distance50.txt",'r')
# distance = eval(dis.readline())

# sta_reward = open("sta_reward.txt",'w')
# true_reward = open("true_reward.txt",'w')

# duplicate = set()

trans = np.ones((config.Map.Width, config.Map.Height, 4), dtype=np.int8)
for i in range(0, config.Map.Width):
    for j in range(0, config.Map.Height):
        if i % 2 == 0:
            trans[i][j][1] = 0
        else:
            trans[i][j][3] = 0
        if j % 2 == 0:
            trans[i][j][2] = 0
        else:
            trans[i][j][0] = 0

agent = Agent()

def get_reward_outside(hole_city):
    agent_gym = AGENT_GYM(config.Map.source_pos, config.Map.hole_pos, config.Game.AgentNum, config.Game.total_time,
                          hole_city, city_dis, trans)
    agent_gym.reset()
    # print mazemap
    r, info = agent.test(agent_gym)
    return r


# model = keras.models.load_model('models/hot_model.h5')
model = EvalNet.build_model()
def get_reward_from_net(hole_city):
    pred_hot, pred_r = model.predict([np.reshape(to_categorical(hole_city, 5), (-1, 20, 5))], batch_size=32)
    return pred_r[0][0]


sim_average = {}
sim_count = 0

class OneAssign:
    def __init__(self, reward_mode=0):
        self.hole_city = []
        self.reward = 0
        self.reward_mode = reward_mode

        self.gen_city()


    def get_reward(self):
        while True:
            have_city = [False for i in range(len(city_dis))]
            for i in self.hole_city:
                have_city[i] = True
            if False in have_city:
                self.gen_city()
            else:
                break
        if self.reward_mode==0:
            self.reward = get_reward_from_net(self.hole_city)
        elif self.reward_mode==1:
            self.reward, info = self.get_reward_from_agent()
            config.train_data.append(info)
        else:
            self.reward, info = self.get_reward_from_agent()
            config.test_data.append(info)

        # self.reward = self.get_reward_from_distance()
        return self.reward

    def get_reward_from_agent(self):
        agent_gym = AGENT_GYM(config.Map.source_pos, config.Map.hole_pos, config.Game.AgentNum, config.Game.total_time,
                              self.hole_city, city_dis, trans)
        agent_gym.reset()
        # print mazemap
        global sim_count
        sim_count += 1

        r, info = agent.test(agent_gym)
        if str(self.hole_city) not in sim_average.keys():
            sim_average[str(self.hole_city)] = [r, 1]
            if r>5550:
                r1, info = agent.test(agent_gym)
                r2, info = agent.test(agent_gym)
                sim_average[str(self.hole_city)] = [r+r1+r2, 3]
                sim_count += 2
        else:
            sim_average[str(self.hole_city)][0] += r
            sim_average[str(self.hole_city)][1] += 1
        return sim_average[str(self.hole_city)][0]/sim_average[str(self.hole_city)][1], info

    def gen_city(self):
        self.hole_city = []
        for i in range(length):
            # self.hole_city.append(np.random.multinomial(1, city_dis, size=1).tolist()[0].index(1))
            self.hole_city.append(random.choice(range(len(city_dis))))

    def update_reward(self):
        if self.reward_mode > 0:
            self.reward = sim_average[str(self.hole_city)][0]/sim_average[str(self.hole_city)][1]


def assign_cmp(a, b):
    if a.reward < b.reward:
        return 1
    elif a.reward == b.reward:
        return 0
    return -1


class Evolution:
    def __init__(self, steps=100, group=None, noble_num=100, civilian_num=1000, output=False, noble_stability=0.5,
                 civilian_random=0.25, extra_civil_evo=1, id=0):
        self.step = 0
        self.mutate_rate = 0.25
        self.noble_num = noble_num
        self.civilian_num = civilian_num
        self.end = False
        self.output = output
        self.noble_stability = noble_stability
        self.civilian_random = civilian_random
        self.extra_civil_evo = extra_civil_evo
        self.time_stamp = time.time()
        if group is None:
            self.noble_group = [OneAssign(1) for _ in range(self.noble_num)]
        else:
            self.noble_group = group + [OneAssign(1) for _ in range(self.noble_num)]
        self.civilian_group = [OneAssign(0) for _ in range(self.civilian_num)]
        self.end_step = steps
        for assign in self.noble_group:
            assign.get_reward()
        for assign in self.civilian_group:
            assign.get_reward()
        r_sum = 0
        for g in self.noble_group:
            r_sum += g.reward
        print r_sum / 100

        self.id = id

        if output:
            self.sim_reward = open("sim_reward_"+str(self.id)+".txt", 'w')
            self.best_assign = open("best_assign_"+str(self.id)+".txt", 'w')

        self.current_best = None

    def reproduce(self):
        self.step += 1
        self.duplicate = set()

        # new noble
        new_noble = []
        for assign in self.noble_group:
            assign.update_reward()
            self.duplicate.add(str(assign.hole_city))

        for i in range(int(2*self.noble_num*self.noble_stability)):
            new_noble.append(self.cross(1, 0))
            new_noble[i].get_reward()
            self.duplicate.add(str(new_noble[i].hole_city))

        # global model
        # model = EvalNet.train_model(model, config.data, 75, False)

        new_civilian = []
        for i in range(int(2*self.civilian_num*(1-self.civilian_random))):
            new_civilian.append(self.cross(0, 1))
            new_civilian[-1].get_reward()
            self.duplicate.add(str(new_civilian[-1].hole_city))

        self.civilian_group += new_civilian
        self.civilian_group.sort(key=cmp_to_key(assign_cmp))
        self.civilian_group = self.civilian_group[:self.civilian_num]

        # extra civilian evo rounds
        for i in range(self.extra_civil_evo):
            new_civilian = []
            for j in range(int(2*self.civilian_num*(1-self.civilian_random))):
                new_civilian.append(self.cross(0, 1))
                new_civilian[-1].get_reward()
                self.duplicate.add(str(new_civilian[-1].hole_city))
            self.civilian_group += new_civilian
            self.civilian_group.sort(key=cmp_to_key(assign_cmp))
            self.civilian_group = self.civilian_group[:self.civilian_num]

        ingoing_noble = self.civilian_group[:int(2*self.noble_num*(1-self.noble_stability))]
        for assign in ingoing_noble:
            assign.reward_mode = 2
            assign.get_reward()
            assign.reward_mode = 1
        ingoing_noble.sort(key=cmp_to_key(assign_cmp))  # record top ingoing noble

        # EvalNet.eval_current_model(model, config.data[-len(ingoing_noble):])

        self.noble_group += new_noble+ingoing_noble
        self.noble_group.sort(key=cmp_to_key(assign_cmp))
        declining_noble = self.noble_group[self.noble_num:]
        for assign in declining_noble:
            assign.reward_mode = 0
            assign.get_reward()
        self.civilian_group = self.civilian_group[int(2 * self.noble_num * (1 - self.noble_stability)):] + declining_noble
        for j in range(int(2 * self.civilian_num * self.civilian_random)):
            self.civilian_group.append(OneAssign())
            self.civilian_group[-1].get_reward()
            self.duplicate.add(str(self.civilian_group[-1].hole_city))
        self.noble_group = self.noble_group[:self.noble_num]

        for j in range(5):
            self.noble_group[j].get_reward()  # update top noble

        global model
        model = EvalNet.smart_train_model(id=self.id)

        if self.step == self.end_step:
            self.end = True
        # select the bests
        if self.step%1==0:
            print("step: " + str(self.step) + " best: " + str(self.noble_group[0].reward))
            print("2-8")
            for j in range(7):
                print self.noble_group[j+1].reward, self.noble_group[j+1].hole_city


        if self.step%1 == 0 or self.step == 1:
            if self.current_best is None or self.current_best.hole_city != self.noble_group[0].hole_city:
                self.current_best = self.noble_group[0]
                if self.output:
                    self.sim_reward.write(str(1000) + '\n')
            if self.output:
                print("--------------------------------------------------------")
                # true_reward.write(str(self.group[0].get_reward_from_agent())+'\n')
                # sta_reward.write(str(self.group[0].reward) + '\n')
                print self.noble_group[0].hole_city
                print sim_average[str(self.noble_group[0].hole_city)]
                self.sim_reward.write(str([self.noble_group[0].reward, sim_count, (time.time()-self.time_stamp)/60]) + '\n')
                self.best_assign.write(str(self.noble_group[0].reward) + '\n')
                self.best_assign.write(str(self.noble_group[0].hole_city) + '\n')
                # self.blood_log.write(str(self.noble_group[0].blood)+'\t'+str(self.civilian_group[0].blood)+'\t'+
                #                      str(self.civilian_group[1000].blood)+'\t'+str(self.civilian_group[2000].blood)+
                #                      '\t'+str(self.civilian_group[3000].blood)+'\n')
                self.sim_reward.close()
                self.best_assign.close()

                self.sim_reward = open("sim_reward_"+str(self.id)+".txt", 'a')
                self.best_assign = open("best_assign_"+str(self.id)+".txt", 'a')

                # with open("noble_check_point_"+str(self.id)+'.txt', 'w') as cp:
                #     for assign in self.noble_group:
                #         cp.write(str(assign.hole_city) + '\t' + str(assign.reward) + '\n')
                #
                # with open("civil_check_point_"+str(self.id)+'.txt', 'w') as cp:
                #     for assign in self.civilian_group:
                #         cp.write(str(assign.hole_city) + '\t' + str(assign.reward) + '\n')
                #
                # with open("sim_check_point_"+str(self.id)+'.txt', 'w') as cp:
                #     cp.write(str(sim_average) + '\n')


    def cross(self, reward_mode=0, is_civilian=1):
        son = OneAssign(reward_mode)
        for i in range(5):
            cross_pos = random.randint(1, length-2)
            if is_civilian:
                father = self.civilian_group[random.randint(0, len(self.civilian_group) - 1)]
                mother = self.civilian_group[random.randint(0, len(self.civilian_group) - 1)]
            else:
                father = self.noble_group[random.randint(0, len(self.noble_group) - 1)]
                mother = self.noble_group[random.randint(0, len(self.noble_group) - 1)]
            son.hole_city[:cross_pos] = father.hole_city[:cross_pos]
            son.hole_city[cross_pos:] = mother.hole_city[cross_pos:]
            if str(son.hole_city) not in self.duplicate:
                self.duplicate.add(str(son.hole_city))
                break

        mutate = random.random()
        if mutate < self.mutate_rate:
        # if mutate < max([self.mutate_rate, int(0.2 > (self.group[0].reward - self.group[-1].reward))]):
            for i in range(5):
            # for i in range(max([5, 10 * int(0.2 > (self.group[0].reward - self.group[-1].reward))])):
                mutate_pos = random.randint(0, length-1)
                # son.hole_city[mutate_pos] = np.random.multinomial(1, city_dis, size=1).tolist()[0].index(1)
                son.hole_city[mutate_pos] = random.choice(range(len(city_dis)))
        # son.get_reward()
        return son

    def run(self):
        time11 = time.time()
        while not self.end:
            time22 = time.time()
            self.reproduce()
            print "One round:", (time.time() - time22)/60.0
        print "Total:", (time.time() - time11)/3600.0
        # return copy.deepcopy(self.current_best)
        return copy.deepcopy(self.noble_group)


def one_evo(i):
    Evolution(steps=100, group=None, noble_num=100, civilian_num=5000, output=True, noble_stability=0.75,
                 civilian_random=0.25, extra_civil_evo=1, id=i).run()


if __name__ == "__main__":
    # time1 = time.time()
    # pl = mp.Pool(6)
    # for i in range(10, 16, 1):
    #     pl.apply_async(one_evo,[i])
    # pl.close()
    # pl.join()
    one_evo(11)