import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Input, Flatten, Conv2DTranspose, Reshape, Concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K
import keras.callbacks
import os
import time
import copy

from sklearn import linear_model

import agent.result_visualizer as RV
import config
import matplotlib
import matplotlib.pyplot as plt

train_test_split = 10000


def load_data():
    time1 = time.time()
    data_list = [[[] for __ in range(17)] for _ in range(5841)]
    for (root, dirs, files) in os.walk("thers/"):
        for filename in files:
            with open('thers/'+filename,'r') as log:
                split_name = filename.split('_')
                hole = eval(log.readline())
                r = eval(log.readline())
                hot = eval(log.readline())
                target = data_list[eval(split_name[1])-1][eval(split_name[2])]
                target.append(hole)
                target.append(r)
                target.append(hot)
    time2 = time.time()
    print "finish load:", time2-time1
    return data_list

def load_init_data():
    with open("clean_data.txt", 'r') as clean_data:
        data_list = []
        for line in clean_data.readlines():
            data_list.append(eval(line))

        config.data = data_list

def make_data(data):
    time1 = time.time()
    wash = set()
    holes = []
    reward = []
    hots = []
    new_data = []
    for i in range(len(data)-1,-1,-1):
        if str(data[i][0]) not in wash:
            wash.add(str(data[i][0]))
            holes.append(data[i][0])
            reward.append(data[i][1])
            hots.append(data[i][2])
            new_data.append(data[i])
    config.data = new_data

    time2 = time.time()
    print "finish make:", time2 - time1
    # print np.array(hots, dtype=np.float32).shape
    return np.array(holes), np.reshape(np.array(hots, dtype=np.float32)/20,(-1,400)), np.array(reward, dtype=np.float32)/100-40

def make_test_data(data):
    time1 = time.time()
    wash = set()
    holes = []
    reward = []
    hots = []
    for i in range(len(data)-1,-1,-1):
        if str(data[i][0]) not in wash:
            wash.add(str(data[i][0]))
            holes.append(data[i][0])
            reward.append(data[i][1])
            hots.append(data[i][2])

    time2 = time.time()
    # print "finish make:", time2 - time1
    # print np.array(hots, dtype=np.float32).shape
    return np.array(holes), np.reshape(np.array(hots, dtype=np.float32)/20,(-1,400)), np.array(reward, dtype=np.float32)/100-40

def smart_make_data(data):
    time1 = time.time()
    wash = set()
    holes = []
    reward = []
    hots = []
    new_data = []
    for i in range(len(data)-1,-1,-1):
        if str(data[i][0]) not in wash:
            wash.add(str(data[i][0]))
            holes.append(data[i][0])
            reward.append(data[i][1])
            hots.append(data[i][2])
            new_data.append(data[i])

    old_data = config.train_data
    config.train_data = copy.deepcopy(new_data)
    del old_data

    time2 = time.time()
    # print "finish make:", time2 - time1
    # print np.array(hots, dtype=np.float32).shape
    return np.array(holes), np.reshape(np.array(hots, dtype=np.float32)/20,(-1,400)), np.array(reward, dtype=np.float32)/100-40


def wash_test_set(data):
    wash = set()
    holes = []
    reward = []
    hots = []
    new_data = []
    for i in range(len(data) - 1, -1, -1):
        if str(data[i][0]) not in wash:
            wash.add(str(data[i][0]))
            holes.append(data[i][0])
            reward.append(data[i][1])
            hots.append(data[i][2])
            new_data.append(data[i])
    config.test_data = new_data

def my_loss(y_true,y_pred):
    return K.mean((y_pred-y_true),axis = -1)


def build_model():
    K.clear_session()

    hole_input = Input(shape=(20, 5), name='hole_input')

    x = Flatten()(hole_input)
    x = Dense(128, activation='relu')(x)

    x = Dense(400, activation='relu')(x)
    x = Reshape((20, 20, 1))(x)

    # x = Conv2DTranspose(16, (3, 3), padding='same', activation='relu')(x)
    x = Conv2DTranspose(16, (3, 3), padding='same', activation='relu')(x)
    h = Conv2DTranspose(1, (3, 3), padding='same', activation='relu')(x)
    new_hot = Flatten()(h)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    R = Dense(1, activation='relu')(x)

    model = Model(inputs=[hole_input], outputs=[new_hot, R])

    model.compile(optimizer=Adam(),
                  loss='mse',
                  metrics=['mse'])
    return model

def get_model():
    model = keras.models.load_model('models/hot_model.h5')
    return model


def train_model(model, data, epochs=30, last_enhance=False):
    holes, hots, reward = make_data(data)
    # print holes.shape
    # print hots.shape
    print reward.shape
    model = build_model()

    if last_enhance:
        model.fit([np.reshape(to_categorical(holes[-192:], 5), (-1, 20, 5))], [hots[-192:],reward[-192:]], epochs=epochs/10, batch_size=64, verbose=0)
    model.fit([np.reshape(to_categorical(holes, 5),(-1,20,5))], [hots,reward], epochs=epochs, batch_size=64, verbose=0)

    model.save('models/hot_model.h5')
    return model


def smart_train_model(min_epochs=10, max_epochs=200, epoch_step=10, id=0):
    # print config.test_data[-10:]
    # print config.train_data[-10:]
    time1 = time.time()
    wash_test_set(config.test_data)
    test_len = len(config.test_data)
    if test_len > 100:
        config.train_data += config.test_data[:test_len-100]
        config.test_data = config.test_data[-100:]

    holes, hots, reward = smart_make_data(config.train_data)
    test_holes, test_hots, test_reward = make_test_data(config.test_data)
    # print holes.shape
    # print hots.shape
    print reward.shape
    model = build_model()

    corr_history = []
    for i in range(min_epochs, max_epochs+1, epoch_step):
        model.fit([np.reshape(to_categorical(holes, 5),(-1,20,5))], [hots,reward], epochs=epoch_step, batch_size=64, verbose=0)
        pred_hot, pred_r = model.predict([np.reshape(to_categorical(test_holes, 5), (-1, 20, 5))], batch_size=32)
        corr_history.append(np.corrcoef(np.reshape(pred_r, (-1)), test_reward)[0][1])
        if len(corr_history)>2 and corr_history[-1] < corr_history[-2] and corr_history[-3] < corr_history[-2]:
            break
    model.save('models/hot_model'+str(id)+'.h5')
    print "correlation", corr_history
    print "training time", time.time() - time1
    return model


def eval_current_model(model, data):
    holes, hots, reward = make_test_data(data)
    score = model.evaluate([np.reshape(to_categorical(holes, 5), (-1, 20, 5))], [hots,reward], batch_size=32, verbose=0)
    print "Score", score
    pred_hot, pred_r = model.predict([np.reshape(to_categorical(holes, 5), (-1, 20, 5))], batch_size=32)
    print "Correlation:", np.corrcoef(np.reshape(pred_r, (-1)), reward)


def huge_map_city(nb_hole, citydis):
    hole_city = [-1 for i in range(nb_hole)]
    for i in range(nb_hole / 4):
        choice = np.random.randint(0, nb_hole)
        while hole_city[choice] != -1:
            choice = np.random.randint(0, nb_hole)
        hole_city[choice] = i
    for i in range(nb_hole):
        if hole_city[i] == -1:
            hole_city[i] = np.random.multinomial(1, citydis, size=1).tolist()[0].index(1)
    return hole_city

city_dis = [0.36666666666666666, 0.26666666666666666, 0.2, 0.13333333333333333, 0.03333333333333333]


def test_model():
    holes, hots, reward = load_test_set()
    # holes, hots, reward = load_singular_test_set()
    print len(hots)
    model = keras.models.load_model('models/hot_model.h5')
    print model.metrics_names
    score = model.evaluate([np.reshape(to_categorical(holes, 5),(-1,20,5))],[hots, reward], batch_size=32)
    print score
    pred_hot, pred_r = model.predict([np.reshape(to_categorical(holes, 5),(-1,20,5))], batch_size=32)

    print pred_hot.shape
    pred_hot = np.reshape(pred_hot,(1600, 20,20))
    with open('result/preds.txt','w') as p_log:
        p_log.write(str(pred_hot.tolist()) + '\n')
        p_log.write(str(pred_r.tolist()) + '\n')
        p_log.write(str(hots.tolist()) + '\n')
        p_log.write(str(reward.tolist()) + '\n')
    # RV.draw_thermal(20,)

def draw_test(k):
    with open('result/preds.txt','r') as p_log:
        pred_hot = np.array(eval(p_log.readline()))
        pred_r = np.array(eval(p_log.readline()),dtype=np.float32)
        hot = np.array(eval(p_log.readline()), dtype=np.float32)
        r = np.array(eval(p_log.readline()), dtype=np.float32)
        print pred_hot.shape
        print pred_r.shape
        print hot.shape
        print r.shape
        hot = np.reshape(hot,(1600,20,20))
        # pred = pred/np.max(pred)
        # target = target/np.max(target)
    # print pred[k]
    # print target[k]
    for i in range(k):
        print i
        RV.draw_thermal(20, pred_hot[i], str(i)+'_pred', pred_r[i][0])
        RV.draw_thermal(20, hot[i], str(i)+'_target', r[i])

def static_test():
    def get_reward_from_distance(hole_city):
        mins = np.zeros((config.Source_num, len(config.Map.city_dis)))
        for i in range(config.Source_num):
            for j in range(config.Hole_num):
                new_dis = distance[i][j]
                if mins[i][hole_city[j]] == 0 or new_dis < mins[i][hole_city[j]]:
                    mins[i][hole_city[j]] = new_dis
            mins[i] *= np.array(config.Map.city_dis)
        return 999 - np.sum(mins)

    data_list = load_data()
    holes, hots, reward = make_data(data_list)
    dis = open("distance20.txt", 'r')
    distance = eval(dis.readline())
    dis.close()
    y = reward[:train_test_split]
    x = np.zeros((train_test_split, 1))
    test_x = np.zeros((train_test_split, 1))
    for i in xrange(train_test_split):
        x[i][0] = get_reward_from_distance(holes[i])
    for i in xrange(1600):
        test_x[i][0] = get_reward_from_distance(holes[i+train_test_split])
    regr = linear_model.LinearRegression()
    regr.fit(x, y)  # train model
    predict_outcome = regr.predict(test_x)
    print "reward:", reward[train_test_split:train_test_split+100]
    print "pred:", predict_outcome[:100]
    print "static_mse:", np.mean((predict_outcome-reward[train_test_split:train_test_split+1600])**2)


def make_test_set():
    data_list = load_data()
    holes, hots, reward = make_data(data_list)
    with open('result/test_set.txt','w') as p_log:
        p_log.write(str(holes[train_test_split:train_test_split+1600, :].tolist()) + '\n')
        p_log.write(str(hots[train_test_split:train_test_split+1600, :].tolist()) + '\n')
        p_log.write(str(reward[train_test_split:train_test_split+1600].tolist()) + '\n')

def load_test_set():
    with open('result/test_set.txt', 'r') as p_log:
        holes = np.array(eval(p_log.readline()))
        hots = np.array(eval(p_log.readline()))
        reward = np.array(eval(p_log.readline()))
    return holes, hots, reward

def load_singular_test_set():
    with open('result/test_set.txt', 'r') as p_log:
        holes = np.array(eval(p_log.readline()))
        hots = np.array(eval(p_log.readline()))
        reward = np.array(eval(p_log.readline()))
    holes_s = []
    hots_s = []
    reward_s = []
    single_set = set()
    for i in range(1600):
        if str(holes[i]) not in single_set:
            holes_s.append(holes[i])
            hots_s.append(hots[i])
            reward_s.append(reward[i])
            single_set.add(str(holes[i]))

    return np.array(holes_s), np.array(hots_s), np.array(reward_s)

def get_correlation():
    with open('result/preds.txt','r') as p_log:
        pred_hot = np.array(eval(p_log.readline()))
        pred_r = np.array(eval(p_log.readline()),dtype=np.float32)
        hot = np.array(eval(p_log.readline()), dtype=np.float32)
        r = np.array(eval(p_log.readline()), dtype=np.float32)
        print pred_r.shape
        print r.shape
    print np.corrcoef(np.reshape(pred_r,(1600)), r)

def draw_correlation():
    with open('result/preds.txt','r') as p_log:
        pred_hot = np.array(eval(p_log.readline()))
        pred_r = np.array(eval(p_log.readline()),dtype=np.float32)
        hot = np.array(eval(p_log.readline()), dtype=np.float32)
        r = np.array(eval(p_log.readline()), dtype=np.float32)
    # plt.plot(pred_r,r)
    plt.scatter(pred_r,r)
    plt.xlim(10,28)
    plt.ylim(10,28)
    plt.show()

# train_model()
# make_test_set()
# test_model()
# draw_test(20)
# static_test()

# a = np.array([41.35,41.01,40.96,40.62,40.67,40.64,41.06,41.00,41.26,40.73])
# a /= 2
# print np.mean((a-np.mean(a))**2)

# 0.982537950046
# get_correlation()
# draw_correlation()
if __name__ == "__main__":
    a = [0] *10
    a[0]+=1
    print a
