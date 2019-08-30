import numpy as np
import config
import os
from keras.layers import Layer
import tensorflow as tf
from collections import deque

# right, down, up, left
dirs = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
dir_symbols = ['>', 'v', '^', '<']
#map_symbols = ['0', '1', '2', '3']
map_symbols = ['.', '#', 'S', 'T']


def inMap(pos):
    [x, y] = pos
    return x >= 0 and x < config.Map.Width and y >= 0 and y < config.Map.Height

def initMazeMap():
    mazemap = np.zeros([config.Map.Width, config.Map.Height, 4], dtype=np.int64)
    return mazemap

def displayMap(mazemap):
    output = ''
    for i in range(config.Map.Height):
        for j in range(config.Map.Width):
            cell = mazemap[i, j]
            for k in range(4):
                if cell[k]:
                    output += map_symbols[k]
        output += '\n'
    print(output)

def displayQvalue(qvalues):
    if len(qvalues)==config.Map.Height*config.Map.Width+1:
        idx = 0
        output = ''
        for i in range(config.Map.Height):
            for j in range(config.Map.Width):
                output += '%7.3f  '%(qvalues[idx])
                idx += 1
            if i==config.Map.Height-1:
                output += '%7.3f  '%(qvalues[-1])
            output += '\n'
        print(output)
    else:
        output = ''
        for i in range(len(qvalues)):
            output += '%7.3f  ' % (qvalues[i])
        print(output)

def remove(path):
    if os.path.exists(path):
        os.remove(path)

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def removedirs(path):
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)

def get_tau(reward_for_prob_one_of_ten):
    return reward_for_prob_one_of_ten / -np.log(0.1)

def get_session():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


class qlogger(object):
    def __init__(self):

        self.minq = 1e20
        self.maxq = -1e20
        self.pre_minq = 1e20
        self.pre_maxq = -1e20
        self.cur_minq = 1e20
        self.cur_maxq = 1e20
        self.mean_maxq = deque(maxlen=1000)


def string_values(list, format='%.3f '):
    output = ''
    for x in list:
        output += format % x
    return output