# SHMetro

---

**Rebuild from**

Metro Crowd Flow from Shanghai Metro System. ([PVCGN](https://github.com/HCPLab-SYSU/PVCGN))

## Info

- Time span: 7/1/2016 - 9/30/2016, 05:30~23:30, 73 intervals
- Number of stations: 288
- Interval: 15min
- Feature: inflow, outflow

## Data Description

- `vel_x.pth, vel_y.pth`: torch tensor
  ```
  shape (batch, channel, n_his or n_pred, n_vertex), (Batch, 2, 4, 288)
  channel_0, inflow
  channel_1, outflow
  ```

- `data_time_#.h5`: pandas table, timestamp for all data, (Batch, 4)
  ```
                       0  ...                   3
  0    2016-07-01 05:30:00  ... 2016-07-01 06:15:00
  1    2016-07-01 05:45:00  ... 2016-07-01 06:30:00
  2    2016-07-01 06:00:00  ... 2016-07-01 06:45:00
  3    2016-07-01 06:15:00  ... 2016-07-01 07:00:00
  4    2016-07-01 06:30:00  ... 2016-07-01 07:15:00
  ```
  
- `time_index_#.pth`: torch long tensor, time index for input data
  ```
  shape (batch, channel, n_his), (6072, 2, 4)
  channel_0, dayofweek, (0-7)
  channel_1, timeofday, (0-73), (0-num time in one day)
  ```
  
- `adj.pth`: torch tensor, 0-1 connectivity graph with self loop  
  ```
  shape (n_vertex, n_vertex), (288, 288)
  total 248 directed edges (including self loops; undirected graph)
  ```
  
- `distance.pth`: torch tensor, distance graph calculate by using 0-1 connectivity graph(adj.pth)
with setting distance between vertex as 1.
  ```
  shape (n_vertex, n_vertex), (288, 288)
  ```
  
- `k_neighbor.pth`: torch tensor, top k% nearest neighbor graph base on distance.pth
  ```
  shape (n_vertex, n_vertex), (288, 288)
  Values:
  Top 10% set to 0,
  Top 10%-20% set to 1, 
  ...
  Top 90%-100% set to 9.
  ```
  
## Load Data

```python
import torch
import numpy as np
import pandas as pd

x = torch.load('vel_x.pth')
y = torch.load('vel_y.pth')

timestamp = pd.read_hdf('data_time.h5')

time_index = np.load('time_index_x.npy')

adj = torch.load('adj.pth')

print(x.shap, y.shape, timestamp.shape, time_index.shape, adj.shape)
```
