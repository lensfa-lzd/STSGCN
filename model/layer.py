import sys

import torch
import torch.nn as nn

from einops import repeat
from einops.layers.torch import Rearrange


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        x = torch.einsum('bctv->bvtc', x)
        # print(x.shape)
        x = self.linear(x)
        x = torch.einsum('bvtc->bctv', x)

        return x


class FullyConnection(nn.Module):
    """
        Liner layer for everytime step.
        To compute the FC operation in parallel.
    """

    def __init__(self, n_time, n_vertex, in_features, out_features, device):
        super(FullyConnection, self).__init__()

        self.weight = nn.Parameter(torch.rand((n_time, in_features, out_features), requires_grad=True, device=device))
        self.bias = nn.Parameter(torch.rand(n_time, requires_grad=True, device=device))

        self.register_parameter('weight', self.weight)
        self.register_parameter('bias', self.bias)

        self.n_vertex = n_vertex
        self.out_features = out_features

    def forward(self, x):
        """

        :param x: [batch_size, channel, n_time, n_vertex]
        :return:
        """
        bias = repeat(self.bias, 't -> t v f', v=3*self.n_vertex, f=self.out_features)
        x = torch.einsum('bctv->btvc', x)
        x = torch.einsum('btvc,tcf->btvf', (x, self.weight))
        x = x + bias
        x = torch.einsum('btvf->bftv', x)

        return x


class Gcn(nn.Module):
    """
        y = A * X
        The param will be multiplied in the following FC layer
    """

    def __init__(self, n_channel, activation):
        super(Gcn, self).__init__()

        if activation == 'glu':
            self.liner = nn.Linear(in_features=n_channel, out_features=2*n_channel)
            self.act = nn.GLU(dim=-1)
        elif activation == 'relu':
            self.liner = nn.Linear(in_features=n_channel, out_features=n_channel)
            self.act = nn.ReLU()
        else:
            print('Activation type error')
            sys.exit()

    def forward(self, adj, x):
        """

        :param adj: [n_vertex, n_vertex]
        :param x: [batch_size, channel, 3*n_vertex]
        :return: [batch_size, channel, 3*n_vertex]
        """
        # x = torch.einsum('bcv->bcvt', x)
        # x = torch.einsum('vn,bcnt->bcvt', (adj, x))
        # x = torch.einsum('bcvt->bctv', x)
        x = torch.einsum('bcv->bvc', x)
        x = torch.einsum('vn,bvc->bnc', (adj, x))

        x = self.liner(x)
        x = self.act(x)

        x = torch.einsum('bvc->bcv', x)

        return x


class OutputLayer(nn.Module):
    """
        [batch_size, channel, n_time, n_vertex] ->
        [batch_size, n_vertex, channel*n_time] ->
    """

    def __init__(self, args):
        super(OutputLayer, self).__init__()
        n_channel = args.n_channel
        n_time = args.n_his

        out_channel = args.out_channel
        out_time = args.n_pred

        in_time = n_time - args.n_layer*2

        self.reshape_1 = Rearrange('b c t v -> b v (c t)')
        self.fully_1 = nn.Linear(in_features=in_time * n_channel, out_features=n_channel)
        self.fully_2 = nn.Linear(in_features=n_channel, out_features=out_time * out_channel)
        self.reshape_2 = Rearrange('b v (c t) -> b c t v', c=out_channel)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.reshape_1(x)
        x = self.fully_1(x)
        x = self.act(x)
        x = self.fully_2(x)
        x = self.reshape_2(x)

        return x
