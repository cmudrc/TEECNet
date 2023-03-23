import torch
import torch_geometric.nn as nn
from torch_geometric.nn.unpool import knn_interpolate


class EdgeConv(nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='add')
        self.mlp = torch.nn.Sequential(nn.Linear(2*in_channels, 32), torch.nn.Tanh(), torch.nn.BatchNorm1d(32), torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.BatchNorm1d(32), torch.nn.Linear(32, out_channels))

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j-x_i], dim=1)
        return self.mlp(tmp)


class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k=6):
        super().__init__(in_channels, out_channels)
        self.k = k

    def forward(self, x, batch=None):
        edge_index = nn.knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return super().forward(x, edge_index)
    

class CFDError(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CFDError, self).__init__()
        # self.convs = FlowMLConvolution(in_channels+1, out_channels, 3, [64, 64, 64])
        self.initial_conv = EdgeConv(in_channels, 64)
        self.first_order_conv = EdgeConv(64, 64)

        self.second_order_conv = EdgeConv(64, 64)

        self.fourth_order_convs = torch.nn.ModuleList()
        for i in range(2):
            self.fourth_order_convs.append(EdgeConv(64, 64))

        self.error_contraction_conv = EdgeConv(192, out_channels)

    def forward(self, data):
        coord, edge_index, batch = data.pos, data.edge_index, data.batch
        x = self.initial_conv(coord, edge_index)
        
        x_1 = self.first_order_conv(x, edge_index)

        x_2 = self.second_order_conv(x, edge_index)
        x_2 = torch.mul(x_2, x_2)

        x_4 = None
        for conv in self.fourth_order_convs:
            temp = conv(x, edge_index)
            temp = torch.mul(temp, temp)
            if x_4 is None:
                x_4 = temp
            else:
                x_4 = x_4 + temp

        x = torch.cat([x_1, x_2, x_4], dim=1)
        x = self.error_contraction_conv(x, edge_index)

        return x


class GraphConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConv, self).__init__()
        self.conv = nn.Sequential('x, edge_index', [(nn.SAGEConv(in_channels, 64), 'x, edge_index -> x'), torch.nn.LeakyReLU(0.1), nn.LayerNorm(64), (nn.SAGEConv(64, 64), 'x, edge_index -> x'), torch.nn.LeakyReLU(0.1), nn.LayerNorm(64), (nn.SAGEConv(64, out_channels), 'x, edge_index -> x')])
        
    def forward(self, x, edge_index):
        return self.conv(x, edge_index)
    

class ErrorInterpolate(torch.nn.Module):
    def __init__(self):
        super(ErrorInterpolate, self).__init__()

    def forward(self, x, pos_l, pos_h):
        return knn_interpolate(x, pos_l, pos_h, k=3)
    

class CFDErrorInterpolate(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CFDErrorInterpolate, self).__init__()
        self.error = CFDError(2, 3)
        self.combine = GraphConv(6, 3)
        self.interpolate = ErrorInterpolate()

    def forward(self, data_l, data_h):
        x, edge_index, pos_l = data_l.x, data_l.edge_index, data_l.pos
        pos_h = data_h.pos

        e = self.error(data_l)
        x = torch.cat([x, e], dim=1)
        x = self.combine(x, edge_index)
        x = self.interpolate(x, pos_l, pos_h)
        return x