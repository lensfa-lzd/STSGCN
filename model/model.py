import sys

import torch
import torch.nn as nn

from einops import repeat, rearrange
from model.layer import Linear, Gcn, OutputLayer


class STSGCM(nn.Module):
    """
        Spatial-Temporal Synchronous Graph Convolutional Module
    """

    def __init__(self, args):
        super(STSGCM, self).__init__()

        gcn_num = args.gcn_num
        n_channel = args.n_channel
        activation = args.activation

        self.gcn = nn.ModuleList([Gcn(n_channel, activation) for _ in range(gcn_num)])

    def forward(self, adj, x):
        """

        :param adj:
        :param x: [batch_size, channel, 3, n_vertex]
        :return: [batch_size, channel, 1, n_vertex]
        """
        x = rearrange(x, 'b c t v -> b c (t v)')
        gcn_output = []
        for gcn in self.gcn:
            # gcn output [batch_size, channel, 3*n_vertex]
            gcn_output.append(gcn(adj, x))

        # Aggregating operation layer
        # [1, batch_size, channel, 3*n_vertex]
        x = torch.stack(gcn_output, dim=0)
        x, _ = torch.max(x, dim=0)

        # Cropping operation
        # Remove the before and after step graph
        n_vertex = x.shape[-1]//3
        x = x[..., n_vertex:-n_vertex]

        x = repeat(x, 'b c v -> b c t v', t=1)

        return x


class STSGCL(nn.Module):
    """
        Embedding before every STSGC Layer
    """

    def __init__(self, layer, args):
        super(STSGCL, self).__init__()
        n_channel = args.n_channel
        n_time = args.n_his
        n_vertex = args.n_vertex

        device = args.device

        self.n_time = n_time
        self.layer = layer

        # print(layer)
        layer_time = n_time - 2*(layer-1)

        # temporal_embedding, [batch_size, channel, n_time, 1]
        # spatial_embedding, [batch_size, channel, 1, n_vertex]

        # self.temporal_emb = nn.Parameter(torch.empty((1, n_channel, n_time, 1), requires_grad=True, device=device))
        self.temporal_emb = nn.Parameter(torch.empty((1, n_channel, layer_time, 1), requires_grad=True, device=device))
        self.spatial_emb = nn.Parameter(torch.empty((1, n_channel, 1, n_vertex), requires_grad=True, device=device))
        torch.nn.init.xavier_normal_(self.temporal_emb, gain=0.0003)
        torch.nn.init.xavier_normal_(self.spatial_emb, gain=0.0003)

        # self.stsgc_module = STSGCM(adj, layer, args)
        self.stsgc_module = nn.ModuleList([STSGCM(args) for _ in range(n_time - 2*layer)])

        self.register_parameter('temporal_emb', self.temporal_emb)
        self.register_parameter('spatial_emb', self.spatial_emb)

    def forward(self, adj, x):
        """

        :param adj:
        :param x: [batch_size, channel, n_time, n_vertex]
        :return: [batch_size, channel, n_time-2, n_vertex]
        """
        # print(x.shape, self.temporal_emb.shape, self.spatial_emb.shape)
        # sys.exit()

        x = x + self.temporal_emb + self.spatial_emb

        need_concat = []
        for i in range(self.n_time - 2*self.layer):
            data = x[:, :, i:i+3, :]
            # need_concat output [batch_size, channel, 1, n_vertex]
            need_concat.append(self.stsgc_module[i](adj, data))

        output = torch.cat(need_concat, dim=2)
        return output


class STSGCN(nn.Module):
    def __init__(self, args):
        """
            x_in (x_data, te)
            size of input/x is [batch_size, channel, n_time, n_vertex]
            size of y/target [batch_size, channel, n_time, n_vertex]
            size of ste is [n_time, n_vertex]
        """
        super(STSGCN, self).__init__()
        in_channel = args.in_channel
        n_channel = args.n_channel

        n_layer = args.n_layer
        n_vertex = args.n_vertex

        device = args.device

        adj = args.adj
        adj = repeat(adj, 'n v -> (f1 n) (f2 v)', f1=3, f2=3)
        self.mask = nn.Parameter(torch.empty((3 * n_vertex, 3 * n_vertex), requires_grad=True, device=device))
        torch.nn.init.xavier_normal_(self.mask, gain=0.0003)
        self.register_parameter('mask', self.mask)

        self.in_channel = in_channel
        # Remove connection for t+1 to t-1 step
        self.n_vertex = n_vertex
        self.adj = self.remove(adj)

        self.input_layer = Linear(in_features=in_channel, out_features=n_channel)

        # Spatial-Temporal Synchronous Graph Convolutional Layers
        self.STSGC_layer = nn.ModuleList([STSGCL(i+1, args) for i in range(n_layer)])

        self.output_layer = OutputLayer(args)

    def remove(self, adj):
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if (i < self.n_vertex) and (j >= 2 * self.n_vertex):
                    adj[i, j] = 0
                if (i >= 2 * self.n_vertex) and (j < self.n_vertex):
                    adj[i, j] = 0

        return adj

    def forward(self, x_in):
        """

        :param x_in: [batch_size, channel, n_time, n_vertex]
        :return:
        """
        adj = self.adj * self.mask
        x, te = x_in[:, :self.in_channel, :, :], x_in[:, -2:, :, :]

        x = self.input_layer(x)

        for layer in self.STSGC_layer:
            x = layer(adj, x)

        x = self.output_layer(x)

        return x
