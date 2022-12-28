import sys

import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.sparse as sp


def process_data():
    with open('train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('val.pkl', 'rb') as f:
        val_data = pickle.load(f)
    with open('test.pkl', 'rb') as f:
        test_data = pickle.load(f)

    train_x = train_data['x']
    val_x = val_data['x']
    test_x = test_data['x']

    train_y = train_data['y']
    val_y = val_data['y']
    test_y = test_data['y']

    train_xtime = train_data['xtime']
    val_xtime = val_data['xtime']
    test_xtime = test_data['xtime']

    train_ytime = train_data['ytime']
    val_ytime = val_data['ytime']
    test_ytime = test_data['ytime']

    data_x = torch.cat((torch.from_numpy(train_x), torch.from_numpy(val_x),
                        torch.from_numpy(test_x)), dim=0).float()
    data_y = torch.cat((torch.from_numpy(train_y), torch.from_numpy(val_y),
                        torch.from_numpy(test_y)), dim=0).float()

    # data_x = torch.einsum('btvc->bctv', data_x)
    # data_y = torch.einsum('btvc->bctv', data_y)
    # torch.save(data_x, 'vel_x.pth')
    # torch.save(data_y, 'vel_y.pth')

    time_x = np.concatenate((train_xtime, val_xtime, test_xtime), axis=0)
    #
    time_x_df = pd.DataFrame(time_x)
    # h5_sd = pd.HDFStore('data_time.h5', 'w')
    # h5_sd['data'] = time_x_df
    # h5_sd.close()

    time_x = np.expand_dims(time_x, axis=1)
    # print(time_x.shape, time_x_df.shape)(1650, 4) (1650, 4)
    n_sam = time_x.shape[0]
    timeofday_x = np.zeros((n_sam, 1, 4))
    dayofweek_x = np.zeros((n_sam, 1, 4))

    time_index_x = np.concatenate((timeofday_x, dayofweek_x), axis=1)
    # print(time_index_x.shape) (1650, 2, 4)

    # print(time_x_df.loc[0][0].dayofweek)
    for i in range(time_index_x.shape[0]):
        for j in range(time_index_x.shape[-1]):
            time_ = time_x_df.loc[i][j]
            hour, min = time_.hour, time_.minute
            timeofday = int((60 * hour + min) / 15)

            dayofweek = time_.dayofweek

            time_index_x[i, 0, j] = dayofweek
            time_index_x[i, 1, j] = timeofday

    # torch.save(torch.from_numpy(time_index_x), 'time_index_x.pth')
    #
    # with open('graph_sh_conn.pkl', 'rb') as f:
    #     adj = pickle.load(f)
    # torch.save(torch.from_numpy(adj), 'adj.pth')

# time_index = np.load('time_index_x.npy')
# print(time_index.shape)
# print(time_index[0, :])

# data_time = pd.read_hdf('data_time.h5')
# print(data_time.shape)

# adj = torch.load('adj.pth')
# print(adj)

# data = torch.load('vel_x.pth')
# print(data.shape)


def Dijkstra(G, start):
    # 输入是从 0 开始，所以起始点减 1
    # start = start - 1
    inf = float('inf')
    node_num = len(G)
    # visited 代表哪些顶点加入过
    visited = [0] * node_num
    # 初始顶点到其余顶点的距离
    dis = {node: G[start][node] for node in range(node_num)}
    # parents 代表最终求出最短路径后，每个顶点的上一个顶点是谁，初始化为 -1，代表无上一个顶点
    parents = {node: -1 for node in range(node_num)}
    # 起始点加入进 visited 数组
    visited[start] = 1
    # 最开始的上一个顶点为初始顶点
    last_point = start

    for i in range(node_num - 1):
        # 求出 dis 中未加入 visited 数组的最短距离和顶点
        min_dis = inf
        for j in range(node_num):
            if visited[j] == 0 and dis[j] < min_dis:
                min_dis = dis[j]
                # 把该顶点做为下次遍历的上一个顶点
                last_point = j
        # 最短顶点假加入 visited 数组
        visited[last_point] = 1
        # 对首次循环做特殊处理，不然在首次循环时会没法求出该点的上一个顶点
        if i == 0:
            parents[last_point] = start + 1
        for k in range(node_num):
            if G[last_point][k] < inf and dis[k] > dis[last_point] + G[last_point][k]:
                # 如果有更短的路径，更新 dis 和 记录 parents
                dis[k] = dis[last_point] + G[last_point][k]
                parents[k] = last_point + 1

    return [values for key, values in dis.items()]
    # 因为从 0 开始，最后把顶点都加 1
    # return {key + 1: values for key, values in dis.items()}, {key + 1: values for key, values in parents.items()}


def distance():
    # inf = float('inf')
    # G = [[0, 1, 12, inf, inf, inf],
    #      [inf, 0, 9, 3, inf, inf],
    #      [inf, inf, 0, inf, 5, inf],
    #      [inf, inf, 4, 0, 13, 15],
    #      [inf, inf, inf, inf, 0, 4],
    #      [inf, inf, inf, inf, inf, 0]]
    # dis, parents = Dijkstra(G, 1)
    # print("dis: ", dis)
    # print("parents: ", parents)

    inf = float('inf')
    adj = torch.load('adj.pth').numpy()
    n_vertex = adj.shape[0]
    id = np.eye(n_vertex)
    adj[adj == 0] = inf
    adj = adj - id
    # print(adj)

    distance = np.zeros((n_vertex, n_vertex))
    for i in range(n_vertex):
        dis = Dijkstra(adj, i)
        distance[i] = np.array(dis)

    print(distance[2])
    distance = torch.from_numpy(distance).float()
    torch.save(distance, 'distance.pth')


def k_graph():
    adj = torch.load('distance.pth').numpy()
    n_vertex = adj.shape[0]
    # print(adj[0])

    graph = np.zeros((n_vertex, n_vertex))
    for i in range(n_vertex):
        n_percent = []
        dis = adj[i]
        for j in range(1, 11):
            n_percent.append(np.percentile(dis, int(j*10)))

        for j in range(10):
            if j == 0:
                bottom = 0
            else:
                bottom = n_percent[j-1]
            top = n_percent[j]

            for k in range(n_vertex):
                if bottom <= dis[k] <= top:
                    graph[i, k] = j

    # print(graph[0])
    graph = torch.from_numpy(graph).float()
    torch.save(graph, 'k_neighbor.pth')


# i = 1
# te = np.load('time_index_x.npy')
# te = torch.from_numpy(te).long()
# torch.save(te, 'time_index_x.pth')

# df = pd.read_hdf('data_time.h5')
# print(df)

index = torch.load('time_index_x.pth')
print(index.shape)