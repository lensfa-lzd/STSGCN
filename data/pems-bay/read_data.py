import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.sparse as sp


def calc_time_index(test):
    test['timeofday'] = None

    for i in range(test.shape[0]):
        time_ = test.loc[i][0]
        hour, min = time_.hour, time_.minute
        index = (60 * hour + min) / 5
        # print(hour, min)
        test.timeofday[i] = int(index)

    test['dayofweek'] = test["time_index"].dt.dayofweek
    test["week_name"] = test["time_index"].dt.day_name()
    print(test)

    h5_sd = pd.HDFStore('time_index_.h5', 'w')
    h5_sd['data'] = test
    h5_sd.close()


def pre_data():
    df = pd.read_hdf('pems-bay.h5')

    time = pd.DatetimeIndex(df.index)
    time_index = []

    year = time.year.values
    month = time.month.values
    day = time.day.values
    minute = time.minute.values
    hour = time.hour.values

    for i in range(time.shape[0]):
        index = str(year[i]) + '-' + str(month[i]).zfill(2) + '-' + str(day[i]).zfill(2) + ' '
        index += str(hour[i]).zfill(2) + ':' + str(minute[i]).zfill(2) + ':00'
        # print(hour, min) nd=np.datetime64('2019-01-10')
        time_index.append(np.datetime64(index))

    data = pd.DataFrame({'time_index': time_index})

    calc_time_index(data)


def k_graph():
    adj = torch.load('adj.pth').numpy()
    n_vertex = adj.shape[0]

    graph = np.zeros((n_vertex, n_vertex))
    for i in range(n_vertex):
        n_percent = []
        dis = adj[i]
        for j in range(1, 11):
            n_percent.append(np.percentile(dis, 100-int(j*10)))

        for j in range(10):
            if j == 0:
                top = 1
            else:
                top = n_percent[j-1]
            bottom = n_percent[j]

            for k in range(n_vertex):
                if bottom <= dis[k] <= top:
                    graph[i, k] = j

    # print(graph)
    # graph = torch.from_numpy(graph).float()
    # torch.save(graph, 'k_neighbor.pth')


index = pd.read_hdf('time_index.h5')
print(index)

#                time_index timeofday  dayofweek week_name
# 0     2017-01-01 00:00:00         0          6    Sunday
# 1     2017-01-01 00:05:00         1          6    Sunday
# 2     2017-01-01 00:10:00         2          6    Sunday
# 3     2017-01-01 00:15:00         3          6    Sunday
# 4     2017-01-01 00:20:00         4          6    Sunday
