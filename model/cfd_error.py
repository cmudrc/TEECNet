import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.unpool import knn_interpolate


class ConvResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvResidualBlock, self).__init__()
        self.gcn = pyg_nn.GCNConv(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else torch.nn.Identity()

    def forward(self, x, edge_index):
        identity = self.residual(x)
        x = self.gcn(x, edge_index)
        x = self.norm(x)
        x = self.act(x)
        x = x + identity
        return x
    

class EdgeConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='add')
        self.mlp = nn.Sequential(pyg_nn.Linear(2*in_channels, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Linear(32, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Linear(32, out_channels))

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
        edge_index = pyg_nn.knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return super().forward(x, edge_index)
    

class CFDError(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CFDError, self).__init__()
        # self.convs = FlowMLConvolution(in_channels+1, out_channels, 3, [64, 64, 64])
        self.initial_conv = EdgeConv(in_channels, 64)
        self.first_order_conv = EdgeConv(64, 64)

        self.second_order_conv = EdgeConv(64, 64)

        self.fourth_order_convs = nn.ModuleList()
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


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.layer1 = ConvResidualBlock(input_dim, hidden_dim)
        self.layer2 = ConvResidualBlock(hidden_dim, hidden_dim)
        self.layer3 = ConvResidualBlock(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x1 = self.layer1(x, edge_index)
        x2 = self.layer2(x1, edge_index)
        x3 = self.layer3(x2, edge_index)
        return x1, x2, x3

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.layer1 = ConvResidualBlock(hidden_dim, hidden_dim)
        self.layer2 = ConvResidualBlock(hidden_dim, output_dim)
        self.layer3 = ConvResidualBlock(output_dim, output_dim)

    def forward(self, x, edge_index):
        x1 = self.layer1(x, edge_index)
        x2 = self.layer2(x1, edge_index)
        x3 = self.layer3(x2, edge_index)
        return x3

class EncoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.message_passing1 = pyg_nn.GCNConv(hidden_dim, hidden_dim)
        self.message_passing2 = pyg_nn.GCNConv(hidden_dim, hidden_dim)
        self.message_passing3 = pyg_nn.GCNConv(hidden_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        enc_x1, enc_x2, enc_x3 = self.encoder(x, edge_index)
        
        message_passing_out1 = self.message_passing1(enc_x1, edge_index)
        message_passing_out2 = self.message_passing2(enc_x2, edge_index)
        message_passing_out3 = self.message_passing3(enc_x3, edge_index)
        
        dec_input = message_passing_out1 + message_passing_out2 + message_passing_out3
        dec_output = self.decoder(dec_input, edge_index)
        
        return dec_output
    

class ErrorInterpolate(torch.nn.Module):
    def __init__(self):
        super(ErrorInterpolate, self).__init__()

    def forward(self, x, pos_l, pos_h):
        return knn_interpolate(x, pos_l, pos_h, k=32)
    

class CFDErrorInterpolate(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CFDErrorInterpolate, self).__init__()
        self.error = CFDError(2, 3)
        self.combine = EncoderDecoder(6, 128, 3)
        self.interpolate = ErrorInterpolate()

    def forward(self, data_l, data_h):
        x, edge_index, pos_l = data_l.x, data_h.edge_index, data_l.pos
        pos_h = data_h.pos

        e = self.error(data_l)
        x = torch.cat([x, e], dim=1)
        x = self.interpolate(x, pos_l, pos_h)
        x = self.combine(x, edge_index) 
        return x