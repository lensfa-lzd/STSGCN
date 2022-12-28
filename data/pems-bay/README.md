# Pems-bay

---

**Rebuild from**

https://github.com/hazdzz/STGCN

## Info

- Time span: 1/1/2017 - 5/31/2017, 00:00~23:55, 288 intervals
- Number of stations: 325
- Interval: 5min
- Feature: flow

## Data Description

- `vel.pth`: torch tensor
  ```
  shape (n_time, n_vertex, channel), (52116, 207, 1)
  ```
  
- `time_index.h5`: pandas table, timestamp for all data, (n_time, 4)
  ```
                 time_index timeofday  dayofweek week_name
  0     2017-01-01 00:00:00         0          6    Sunday
  1     2017-01-01 00:05:00         1          6    Sunday
  2     2017-01-01 00:10:00         2          6    Sunday
  3     2017-01-01 00:15:00         3          6    Sunday
  4     2017-01-01 00:20:00         4          6    Sunday
  ```
  
- `adj.pth`: torch tensor, weighted connectivity graph with self loop

  build by using thresholded Gaussian kernel (Shuman et al., 2013)

  $$
    {{\rm{W}}_{ij}} = \exp ( - \frac{{dist{{({v_i},{v_j})}^2}}}{{{\sigma ^2}}})
  $$

  and if dist > threshold, the value is set to 0,
  Threshold is assigned to 0.1 in this case.
  ```
  shape (n_vertex, n_vertex), (325, 325)
  ```
  
## Load Data

```python
import torch
import pandas as pd

data = torch.load('vel.pth')
time_index = pd.read_hdf('time_index.h5')
adj = torch.load('adj.pth')

print(data.shape, time_index.shape, adj.shape)
print(time_index.head())
```

