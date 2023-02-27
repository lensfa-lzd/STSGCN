import numpy as np
import pandas as pd
import torch

dtx = pd.read_hdf('data_time_x.h5')
dty = pd.read_hdf('data_time_y.h5')
#                     0  ...                   3
# 0 2016-07-01 05:30:00  ... 2016-07-01 06:15:00
# 1 2016-07-01 05:45:00  ... 2016-07-01 06:30:00
# 2 2016-07-01 06:00:00  ... 2016-07-01 06:45:00
# 3 2016-07-01 06:15:00  ... 2016-07-01 07:00:00
# 4 2016-07-01 06:30:00  ... 2016-07-01 07:15:00
#
# [6072 rows x 4 columns]
#
# print(dtx.iloc[4*16: 4*18])
# print(dty.iloc[4*16: 4*18])

datax = torch.load('vel_x.pth')
# print(datax.shape) torch.Size([6072, 2, 4, 288])

time_index = dtx.iloc[:, 0]
print(time_index)

x = torch.einsum()
