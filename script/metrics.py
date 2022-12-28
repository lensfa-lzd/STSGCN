import sys

import torch
import numpy as np

'''
    masked version error functions partly base on PVCGN
    https://github.com/HCPLab-SYSU/PVCGN
'''


def calc_rmse(preds, labels):
    return np.sqrt(calc_mse(preds=preds, labels=labels))


def calc_mse(preds, labels):
    with np.errstate(divide='ignore', invalid='ignore'):
        mse = np.square(np.subtract(preds, labels))
    return np.mean(mse)


def calc_mae(preds, labels):
    with np.errstate(divide='ignore', invalid='ignore'):
        mae = np.abs(np.subtract(preds, labels))
    return np.mean(mae)


def calc_mape_masked(preds, labels):
    with np.errstate(divide='ignore', invalid='ignore'):
        # zero value might be slightly change due to Z-score norm
        mask = labels < 10e-5
        # labels[mask] = 1
        # mape = np.abs(np.divide(np.subtract(preds, labels), labels))
        mape = np.abs((preds-labels)/labels)
        mape[mask] = 0
        return np.mean(mape)


# Mask target value 0 out
class Metrics(object):
    """
        size of output is [batch_size, channel, n_time, n_vertex]
        size of target [batch_size, channel, n_pred, n_vertex]
        thus axis 1 is "channel"
    """

    def __init__(self, target, output):
        self.target = target.numpy()
        self.output = output.numpy()
        # if type in ['validation', 'val', 'test']:
        #     self.mode = 'val'
        # else:
        #     self.mode = 'train'

    def rmse(self):
        rmse = torch.as_tensor([
            calc_rmse(self.output[:, i, :, :], self.target[:, i, :, :])
            for i in range(self.output.shape[1])
        ])
        # if self.mode == 'val':
        #     rmse = rmse.mean().item()

        return rmse.mean().item()

    def mae(self):
        mae = torch.as_tensor([
            calc_mae(self.output[:, i, :, :], self.target[:, i, :, :])
            for i in range(self.output.shape[1])
        ])

        return mae.mean().item()

    def mape(self):
        mape = torch.as_tensor([
            calc_mape_masked(self.output[:, i, :, :], self.target[:, i, :, :])
            for i in range(self.output.shape[1])
        ]) * 100

        return mape.mean().item()

    def all(self):
        rmse = self.rmse()
        mae = self.mae()
        mape = self.mape()

        return rmse, mae, mape