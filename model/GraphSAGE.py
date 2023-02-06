import torch
import torch_geometric.nn as nn


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, num_filters):
        super(GraphSAGE, self).__init__()
        # self.conv1 = nn.Sequential('x, edge_index', [(nn.GraphConv(in_channels, num_filters[0]), 'x, edge_index -> x'), torch.nn.LeakyReLU(0.1), torch.nn.BatchNorm1d(num_filters[0])])
        self.conv1 = nn.Sequential('x, edge_index', [(nn.SAGEConv(in_channels, num_filters[0]), 'x, edge_index -> x'), torch.nn.LeakyReLU(0.1)])
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            # self.convs.append(nn.Sequential('x, edge_index', [(nn.GraphConv(num_filters[i], num_filters[i+1]), 'x, edge_index -> x'), torch.nn.LeakyReLU(0.1), torch.nn.BatchNorm1d(num_filters[i+1])]))
            self.convs.append(nn.Sequential('x, edge_index', [(nn.SAGEConv(num_filters[i], num_filters[i+1]), 'x, edge_index -> x'), torch.nn.LeakyReLU(0.1)]))
        self.conv2 = nn.SAGEConv(num_filters[i+1], out_channels)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = self.conv2(x, edge_index)
        # x = nn.global_mean_pool(x, batch)
        return x