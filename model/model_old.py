import torch
import torch.nn as nn

from einops import repeat
from model.layer import Linear, Gcn, OutputLayer


class STSGCM(nn.Module):
    """
        Spatial-Temporal Synchronous Graph Convolutional Module
    """

    def __init__(self, adj, layer, args):
        super(STSGCM, self).__init__()

        gcn_num = args.gcn_num
        n_channel = args.n_channel
        n_time = args.n_his
        n_vertex = args.n_vertex
        activation = args.activation

        device = args.device

        self.adj = adj
        self.gcn = nn.ModuleList([Gcn(n_time - layer*2, n_vertex, n_channel, activation, device) for _ in range(gcn_num)])

    def forward(self, x):
        gcn_output = []
        for gcn in self.gcn:
            gcn_output.append(gcn(self.adj, x))

        # Aggregating operation layer
        # [1, batch_size, channel, n_time, n_vertex]
        x = torch.stack(gcn_output, dim=0)
        x, _ = torch.max(x, dim=0)

        # Cropping operation
        # Remove the before and after step graph
        n_vertex = x.shape[-1]//3
        x = x[..., n_vertex:-n_vertex]

        return x


class STSGCL(nn.Module):
    """
        Embedding before every STSGC Layer

        [batch_size, channel, n_time, n_vertex] ->
        [batch_size, channel, n_time-2, n_vertex*3] ->

    """

    def __init__(self, adj, layer, args):
        super(STSGCL, self).__init__()
        n_channel = args.n_channel
        n_time = args.n_his
        n_vertex = args.n_vertex

        device = args.device

        # temporal_embedding, [batch_size, channel, n_time, 1]
        # spatial_embedding, [batch_size, channel, 1, n_vertex]
        self.temporal_emb = nn.Parameter(torch.rand((1, n_channel, n_time, 1), requires_grad=True, device=device))
        self.spatial_emb = nn.Parameter(torch.rand((1, n_channel, 1, n_vertex), requires_grad=True, device=device))
        # torch.nn.init.xavier_normal_(self.temporal_emb, gain=0.0003)
        # torch.nn.init.xavier_normal_(self.spatial_emb, gain=0.0003)

        self.stsgc_module = STSGCM(adj, layer, args)

        self.register_parameter('temporal_emb', self.temporal_emb)
        self.register_parameter('spatial_emb', self.spatial_emb)

    def forward(self, x):
        x = x + self.temporal_emb + self.spatial_emb

        # Every time step will be concat with its before and after time step,
        # except the fist and the last step.
        x_before = x[:, :, :-2, :]
        x_after = x[:, :, 2:, :]
        x = torch.cat((x_before, x[:, :, 1:-1, :], x_after), dim=-1)
        x = self.stsgc_module(x)

        return x


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
        mask = nn.Parameter(torch.rand((3 * n_vertex, 3 * n_vertex), requires_grad=True, device=device))
        # torch.nn.init.xavier_normal_(mask, gain=0.0003)

        adj = mask * adj

        self.in_channel = in_channel
        # Remove connection for t+1 to t-1 step
        self.n_vertex = n_vertex
        adj = self.remove(adj)

        self.input_layer = Linear(in_features=in_channel, out_features=n_channel)

        # Spatial-Temporal Synchronous Graph Convolutional Layers
        self.STSGC_layer = nn.ModuleList([STSGCL(adj, i+1, args) for i in range(n_layer)])

        self.output_layer = OutputLayer(args)

        self.register_parameter('mask', mask)

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
        x, te = x_in[:, :self.in_channel, :, :], x_in[:, -2:, :, :]

        x = self.input_layer(x)

        for layer in self.STSGC_layer:
            x = layer(x)

        x = self.output_layer(x)

        return x
