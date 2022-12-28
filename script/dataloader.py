import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from einops import repeat


def load_adj(dataset_name):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    adj = torch.load(os.path.join(dataset_path, 'adj.pth'))
    adj = adj.numpy()

    if dataset_name == 'metr-la':
        n_vertex = 207
    elif dataset_name == 'pems-bay':
        n_vertex = 325
    elif dataset_name == 'hz-metro':
        n_vertex = 80
    elif dataset_name == 'sh-metro':
        n_vertex = 288

    id = np.eye(n_vertex)

    # The "original" adjacency matrix (adj) should NOT be have a self-loop/self-connection (value 1 fill in diagonal)
    # However all adjs in dataset above have a self-loop/self-connection
    # So removing the self-connection here makes it equal to the adjacency matrix (W) in the paper
    adj = adj - id

    adj[adj > 0] = 1  # 0/1 matrix, symmetric

    # adj is weighted in dataset above
    return adj, n_vertex


def load_data(dataset_name, len_train, len_val):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)

    # shape of vel [num_of_data, num_vertex, channel]
    vel = torch.load(os.path.join(dataset_path, 'vel.pth'))
    n_vertex = vel.shape[1]

    # shape of time_index [num_of_data, 2]
    # time_index [..., 0] for dayofweek
    # time_index [..., 1] for timeofday
    time_index = pd.read_hdf(os.path.join(dataset_path, 'time_index.h5'))
    time_index = calc_te(time_index)

    # shape of te [num_of_data, num_vertex, 2]
    # te [..., 0] for dayofweek
    # te [..., 1] for timeofday
    te = repeat(time_index, 't c -> t v c', v=n_vertex)

    train = (vel[: len_train], te[: len_train])
    val = (vel[len_train: len_train + len_val], te[len_train: len_train + len_val])
    test = (vel[len_train + len_val:], te[len_train + len_val:])
    return train, val, test


def load_data_metro(dataset_name, len_train, len_val, n_pred):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)

    # shape of vel (batch, channel, n_his or n_pred, n_vertex)
    vel_x = torch.load(os.path.join(dataset_path, 'vel_x.pth'))
    vel_y = torch.load(os.path.join(dataset_path, 'vel_y.pth'))
    n_vertex = vel_x.shape[-1]

    # time_index
    # shape (batch, channel, n_his), (1650, 2, 4)
    # channel_0, dayofweek, (0 - 7)
    # channel_1, timeofday, (0 - 73), (0 - num time in one day)
    time_index = torch.load(os.path.join(dataset_path, 'time_index_x.pth'))

    # shape of te (batch, channel, n_his, n_vertex)
    # te [..., 0] for dayofweek
    # te [..., 1] for timeofday
    te = repeat(time_index, 'b c t -> b c t v', v=n_vertex)

    train_x = (vel_x[: len_train], te[: len_train])
    train_y = (vel_y[: len_train, :, :n_pred, :],)
    val_x = (vel_x[len_train: len_train + len_val], te[len_train: len_train + len_val])
    val_y = (vel_y[len_train: len_train + len_val, :, :n_pred, :],)
    test_x = (vel_x[len_train + len_val:], te[len_train + len_val:])
    test_y = (vel_y[len_train + len_val:, :, :n_pred, :],)
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def calc_te(time_index):
    # shape of te dayofweek/timeofday [num_of_data, 1]
    dayofweek = time_index.dayofweek.values
    timeofday = time_index.timeofday.values.astype(np.int64)
    dayofweek, timeofday = torch.from_numpy(dayofweek), torch.from_numpy(timeofday)
    dayofweek, timeofday = dayofweek.unsqueeze(dim=-1), timeofday.unsqueeze(dim=-1)
    # timeofday = F.one_hot(timeofday, num_classes=n_timeofday)
    # dayofweek = F.one_hot(dayofweek, num_classes=n_dayofweek)
    return torch.cat((dayofweek, timeofday), dim=-1)


def data_transform(data, n_his, n_pred):
    # produce data slices for x_data and y_data

    # shape of data [num_of_data, num_vertex, channel]
    channel = data.shape[-1]
    n_vertex = data.shape[1]
    len_record = data.shape[0]
    num = len_record - n_his - n_pred

    x = torch.zeros([num, n_his, n_vertex, channel])
    y = torch.zeros([num, n_pred, n_vertex, channel])

    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail]
        y[i, :, :, :] = data[tail: tail + n_pred]

    x = torch.einsum('btvc->bctv', x).float()
    y = torch.einsum('btvc->bctv', y).float()

    # size of input/x is [batch_size, channel, n_time, n_vertex]
    # size of y/target [batch_size, channel, n_time, n_vertex]
    return x, y