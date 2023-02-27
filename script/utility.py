import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from einops import rearrange


class StandardScaler:
    """
        input/output shape [num_of_data, num_vertex, channel]
        or [batch_size, channel, n_time, n_vertex]
    """
    def __init__(self, fit_data):
        if len(fit_data.shape) != 3:
            # shape of fit_data [num_of_data, num_vertex, channel]
            fit_data = rearrange(fit_data, 'd t v c -> (d t) v c')

        fit_data = rearrange(fit_data, 't v c -> (t v) c')
        self.mean = torch.mean(fit_data, dim=0)
        self.std = torch.std(fit_data, dim=0)

    def transform(self, x):
        if len(x.shape) == 3:
            # shape of fit_data [num_of_data, num_vertex, channel]
            v = x.shape[1]
            x = rearrange(x, 't v c -> (t v) c')
            x = (x-self.mean)/self.std
            return rearrange(x, '(t v) c -> t v c', v=v).float()
        else:
            # for metro data [day, n_time, n_vertex, channel]
            v = x.shape[-2]
            batch_size = x.shape[0]
            x = rearrange(x, 'd t v c -> (d t v) c')
            x = (x - self.mean) / self.std

            return rearrange(x, '(d t v) c -> d t v c', d=batch_size, v=v).float()

    def inverse_transform(self, x):
        if len(x.shape) == 3:
            # dataset data arrange by [time, vertex, channel]
            v = x.shape[1]
            x = rearrange(x, 't v c -> (t v) c')
            x = x*self.std + self.mean
            return rearrange(x, '(t v) c -> t v c', v=v)
        else:
            # network output data/target data arrange by [batch_size, channel, time, vertex]
            v = x.shape[-1]
            batch_size = x.shape[0]
            x = rearrange(x, 'b c t v -> (b t v) c')
            x = x * self.std + self.mean
            return rearrange(x, '(b t v) c -> b c t v', b=batch_size, v=v)

    def data_info(self):
        return self.mean, self.std


def calc_eigenmaps(adj, k, padded_vertex):
    """
    # makes difference when adj is directed, used to average the different value between (i->j) and (j->i)
    # but in this case (data in the data folder) all the adjs are undirected, so you can take adj = dir_adj
    adj = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)

    row, col = adj.nonzero()
    adj[row, col] = adj[col, row] = 1  # 0/1 matrix, symmetric
    """

    n_components = connected_components(sp.csr_matrix(adj), directed=False, return_labels=False)
    assert n_components == 1  # the graph should be connected, nonzero value should be 1

    n_vertex = adj.shape[0]

    D = np.sum(adj, axis=1) ** (-1 / 2)
    L = np.eye(n_vertex) - (adj * D).T * D  # normalized Laplacian

    _, v = eigh(L)
    eigenmaps = v[:, 1:(k + 1)]  # eigenvectors corresponding to the k smallest non-trivial eigenvalues

    eigenmaps = eigenmaps.astype(dtype=np.float32)
    eigenmaps = torch.from_numpy(eigenmaps).transpose(-1, -2)
    # shape of eigenmaps [k, n_vertex]

    n_vertex = eigenmaps.shape[-1]
    assert n_vertex <= padded_vertex
    if eigenmaps.shape[-1] != padded_vertex:
        eigenmaps = torch.cat((eigenmaps, eigenmaps[:, :padded_vertex-n_vertex]), dim=-1)
    return eigenmaps


def calc_mask(args, device):
    """
    def CalcCausalMask(n_time):
        mask = torch.triu(torch.ones(n_time, n_time), diagonal=1).float()
        return mask*(-10e5)

    def CalcCausalMask(self):
        n_time = self.norm_shape[1]
        mask_size = n_time*self.windows_size
        mask = torch.zeros(mask_size, mask_size).float()
        for i in range(mask_size):
            moment = i // self.windows_size + 1
            mask_boundary = moment * self.windows_size
            for j in range(mask_size):
                if j > mask_boundary:
                    mask[i, j] = 1

    return mask * (-10e5)
    """
    n_time = args.n_his
    k_window = args.k_window
    n_vertex = args.n_vertex
    dataset_name = args.dataset
    k = args.k_neighbor

    temporal_mask = torch.ones(n_time, n_time).float().to(device)
    temporal_mask = torch.triu(temporal_mask, diagonal=1 + k_window)
    temporal_mask = -10e5 * temporal_mask

    window_size = args.window_size
    mask_size = n_time * window_size
    spatial_mask = torch.zeros(mask_size, mask_size).float().to(device)
    for i in range(mask_size):
        moment = i // window_size + 1 + k_window
        mask_boundary = moment * window_size
        for j in range(mask_size):
            if j > mask_boundary:
                spatial_mask[i, j] = 1

    spatial_mask = -10e5 * spatial_mask

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    k_neighbor = torch.load(os.path.join(dataset_path, 'k_neighbor.pth'))
    neighbor_mask = torch.zeros(n_vertex, n_vertex).float().to(device)
    neighbor_mask[k_neighbor > k] = 1

    neighbor_mask = -10e5 * neighbor_mask
    return temporal_mask, spatial_mask, neighbor_mask


def ShiftPad(x, pad_size):
    """
        x [batch_size, channel, n_time, n_vertex]
        To make n_vertex is a multiple of 4*window_size
        padding in a cycling manner
    """

    return torch.cat((x, x[..., :pad_size]), dim=-1)


def calc_metric(y, y_pred, zscore):
    """
        size of y/y_pred [batch_size, channel, n_pred, n_vertex]
    """
    y, y_pred = y.detach().cpu(), y_pred.detach().cpu()

    y = zscore.inverse_transform(y)
    y_pred = zscore.inverse_transform(y_pred)

    metric = Metrics(y, y_pred)
    # metric = EvaluationMetrics(y, y_pred)
    return metric.all()


class MAELoss(torch.nn.Module):
    """
        size of x/input is [batch_size, channel, n_time, n_vertex]
        size of y/output/target [batch_size, channel, n_time, n_vertex]
        Calc MAE by channel
    """
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x, y):
        # print(x.shape, y.shape)
        # print(self.mean, self.std)
        # metro torch.Size([8, 2, 4, 80]) torch.Size([8, 2, 4, 80]) mean [2] std [2]
        # road torch.Size([8, 1, 12, 325]) torch.Size([8, 1, 12, 325]) mean [1] std [1]
        x, y = torch.einsum('bctv->btvc', x), torch.einsum('bctv->btvc', y)
        x = x * self.std + self.mean
        y = y * self.std + self.mean
        mae = torch.absolute(x-y)
        return torch.mean(mae)

    @staticmethod
    def calc_mae(x, y):
        """
        size of x/input is [batch_size, n_time, n_vertex]
        size of y/output/target [batch_size, n_time, n_vertex]
        """
        mae = torch.absolute(x-y)
        return torch.mean(mae)


# Mask target value 0 out
class Metrics(object):
    """
        masked version error functions partly base on PVCGN
        https://github.com/HCPLab-SYSU/PVCGN

        size of output is [batch_size, channel, n_time, n_vertex]
        size of target [batch_size, channel, n_pred, n_vertex]
        thus axis 1 is "channel"
    """
    def __init__(self, target, output):
        self.target = target
        self.output = output

        # zero value might be slightly change due to Z-score norm
        self.mask = target < 10e-5

    def mse(self):
        mse = torch.square(self.target - self.output)
        return torch.mean(mse)

    def rmse(self):
        return torch.sqrt(self.mse())

    def mae(self):
        return torch.mean(torch.absolute(self.target - self.output))

    def mape(self):
        mape = torch.absolute((self.target - self.output)/self.target)
        mape[self.mask] = 0
        return torch.mean(mape * 100)

    def all(self):
        rmse = self.rmse().item()
        mae = self.mae().item()
        mape = self.mape().item()

        return rmse, mae, mape
