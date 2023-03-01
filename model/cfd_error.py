import torch
import torch_geometric.nn as nn
# from torch_geometric.nn.unpool import knn_interpolate


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
        
        self.edge_conv1 = EdgeConv(2, 64)
        self.edge_convs = torch.nn.ModuleList()
        for i in range(3):
            self.edge_convs.append(EdgeConv(64, 64))
        self.edge_conv2 = EdgeConv(64 * (i + 2), 64)
        self.edge_conv3 = EdgeConv(64, 3)

        self.conv4 = nn.Sequential('x, edge_index', [(nn.SAGEConv(in_channels, 64), 'x, edge_index -> x'), torch.nn.LeakyReLU(0.1)])
        self.convs2 = torch.nn.ModuleList()
        for i in range(3):
            # self.convs.append(nn.Sequential('x, edge_index', [(nn.GraphConv(num_filters[i], num_filters[i+1]), 'x, edge_index -> x'), torch.nn.LeakyReLU(0.1), torch.nn.BatchNorm1d(num_filters[i+1])]))
            self.convs2.append(nn.Sequential('x, edge_index', [(nn.SAGEConv(64, 64), 'x, edge_index -> x'), torch.nn.LeakyReLU(0.1)]))
        self.conv5 = nn.SAGEConv(64, out_channels)

        
    def forward(self, data):
        u, coord, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        x = self.edge_conv1(coord, edge_index)
        append = x
        for conv in self.edge_convs:
            torch.mul(x, x)
            x = conv(x, edge_index)
            append = torch.cat([append, x], dim=1)
        x = self.edge_conv2(append, edge_index)
        x = self.edge_conv3(x, edge_index)


        u = self.conv4(torch.add(u, x), edge_index)
        for conv in self.convs2:
            u = conv(u, edge_index)
        u = self.conv5(u, edge_index)

        return u
