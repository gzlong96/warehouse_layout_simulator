import config
import numpy as np
import queue

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

trans_to_action = [[1,0],[0,1],[-1,0],[0,-1]]

source_pos = config.Map.source_pos
hole_pos = config.Map.hole_pos

def inMap(pos):
    [x, y] = pos
    return x >= 0 and x < config.Map.Width and y >= 0 and y < config.Map.Height

def get_map():
    map = np.zeros((config.Map.Width, config.Map.Height))
    for source in source_pos:
        map[source[0]][source[1]] = -1
    for i in range(len(hole_pos)):
        map[hole_pos[i][0]][hole_pos[i][1]] = 1
    return map


def all_source_dis():
    source_dis = np.zeros((len(source_pos), len(hole_pos)))
    distance = -np.ones((len(source_pos),config.Map.Width, config.Map.Height))
    que = queue.Queue(maxsize=config.Map.Width * config.Map.Height)
    for i in range(len(source_pos)):
        print("source: " + str(i))
        que.put(source_pos[i])
        distance[i][source_pos[i][0]][source_pos[i][1]] = 0
        while not que.empty():
            current = que.get()
            for j in range(4):
                if trans[current[0]][current[1]][j] == 1:
                    next_pos = [current[0] + trans_to_action[j][0], current[1] + trans_to_action[j][1]]
                    if inMap(next_pos) and distance[i][next_pos[0]][next_pos[1]] == -1:
                        distance[i][next_pos[0]][next_pos[1]] = distance[i][current[0]][current[1]] + 1
                        que.put(next_pos)
                        if hs_map[next_pos[0]][next_pos[1]] == 1:
                            index = hole_pos.index(next_pos)
                            # print(str(distance[i][next_pos[0]][next_pos[1]]))
                            # print(str(h_dis[index]))
                            source_dis[i][index] = distance[i][next_pos[0]][next_pos[1]] + h_dis[index]
    return source_dis


def all_hole_dis():
    hole_dis = 9999 * np.ones((len(hole_pos)))
    distance = -np.ones((len(hole_pos),config.Map.Width, config.Map.Height))
    que = queue.Queue(maxsize=config.Map.Width * config.Map.Height)
    for i in range(len(hole_pos)):
        print("hole: "+str(i))
        que.put(hole_pos[i])
        distance[i][hole_pos[i][0]][hole_pos[i][1]] = 0
        while not que.empty():
            current = que.get()
            for j in range(4):
                if trans[current[0]][current[1]][j] == 1:
                    next_pos = [current[0] + trans_to_action[j][0], current[1] + trans_to_action[j][1]]
                    if inMap(next_pos) and distance[i][next_pos[0]][next_pos[1]] == -1:
                        distance[i][next_pos[0]][next_pos[1]] = distance[i][current[0]][current[1]] + 1
                        que.put(next_pos)
                        if hs_map[next_pos[0]][next_pos[1]] == -1:
                            hole_dis[i] = distance[i][next_pos[0]][next_pos[1]]
                            while not que.empty():
                                current = que.get()
    return hole_dis


hs_map = get_map()
h_dis = all_hole_dis()
result = all_source_dis()
print(result)
dis = open("distance50.txt", 'w')
dis.write(str(result.tolist()))