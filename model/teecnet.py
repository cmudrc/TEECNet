import torch
from torch_geometric.utils import add_self_loops, degree
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.nn.inits import reset, uniform
from torch_geometric.nn import EdgeConv, knn_graph
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_scatter import scatter_softmax
from model.neural_operator import KernelNN


class TEECNet(torch.nn.Module):
    r"""The Taylor-series Expansion Error Correction Network which consists of several layers of a Taylor-series Error Correction kernel.

    Args:
        in_channels (int): Size of each input sample.
        width (int): Width of the hidden layers.
        out_channels (int): Size of each output sample.
        num_layers (int): Number of layers.
        **kwargs: Additional arguments of :class:'torch_geometric.nn.conv.MessagePassing'
    """
    def __init__(self, in_channels, width, out_channels, num_layers=4):
        super(TEECNet, self).__init__()
        self.num_layers = num_layers

        self.fc1 = nn.Linear(in_channels, width)
        self.kernel = KernelConv(width, width, kernel=PowerSeriesKernel, in_edge=1, num_layers=2, num_powers=4)
        self.fc_out = nn.Linear(width, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.fc1(x)
        for i in range(self.num_layers):
            x = F.relu(self.kernel(x, edge_index, edge_attr))
        x = self.fc_out(x)
        return x


class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x
    

class PowerSeriesConv(nn.Module):
    def __init__(self, in_channel, out_channel, num_powers, **kwargs):
        super(PowerSeriesConv, self).__init__()
        self.num_powers = num_powers
        self.convs = torch.nn.ModuleList()
        for i in range(num_powers):
            self.convs.append(nn.Linear(in_channel, out_channel))
        self.activation = nn.LeakyReLU(0.1)
        self.root_param = nn.Parameter(torch.Tensor(num_powers, out_channel))

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_powers):
            reset(self.convs[i])
        size = self.num_powers
        uniform(size, self.root_param)

    def forward(self, x):
        x_full = None
        for i in range(self.num_powers):
            x_conv = self.convs[i](x)
            if i == 0:
                x_full = self.root_param[i] * x_conv
            else:
                x_conv = self.root_param[i] * torch.pow(self.activation(x_conv), i)
                x_full = x_conv + x_full
        return x_full
    

class PowerSeriesKernel(nn.Module):
    def __init__(self, num_layers, num_powers, activation=nn.ReLU, **kwargs):
        super(PowerSeriesKernel, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.conv0 = PowerSeriesConv(kwargs['in_channel'], 64, num_powers)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(PowerSeriesConv(64, 64, num_powers))
        self.norm = nn.BatchNorm1d(64)

        self.conv_out = PowerSeriesConv(64, kwargs['out_channel'], num_powers)
        self.activation = activation()

    def forward(self, edge_attr):
        x = self.conv0(edge_attr)
        for i in range(self.num_layers):
            # x = self.activation(self.convs[i](x))
            x = self.convs[i](x)
            x = self.norm(x)
        x = self.conv_out(x)
        return x


class KernelConv(pyg_nn.MessagePassing):
    r"""
    The continuous kernel-based convolutional operator from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.
    This convolution is also known as the edge-conditioned convolution from the
    `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
    Graphs" <https://arxiv.org/abs/1704.02901>`_ paper (see
    :class:`torch_geometric.nn.conv.ECConv` for an alias):

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.*
    a MLP. In our implementation the kernel is combined via a Taylor expansion of 
    graph edge attributes :math:`\mathbf{e}_{i,j}` and a typical neural operator implementation
    of a DenseNet kernel.

    Args:
        in_channel (int): Size of each input sample (nodal values).
        out_channel (int): Size of each output sample (nodal values).
        kernel (torch.nn.Module): A kernel function that maps edge attributes to
            edge weights.
        in_edge (int): Size of each input edge attribute.
        num_layers (int): Number of layers in the Taylor-series expansion kernel.
    """
    def __init__(self, in_channel, out_channel, kernel, in_edge=1, num_layers=3, **kwargs):
        super(KernelConv, self).__init__(aggr='mean')
        self.in_channels = in_channel
        self.out_channels = out_channel
        self.in_edge = in_edge
        self.root_param = nn.Parameter(torch.Tensor(in_channel, out_channel))
        self.bias = nn.Parameter(torch.Tensor(out_channel))

        self.kernel = kernel(in_channel=in_edge, out_channel=out_channel**2, num_layers=num_layers, **kwargs)
        self.operator_kernel = DenseNet([in_edge, 128, 128, out_channel**2], nn.ReLU)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.kernel)
        reset(self.operator_kernel)
        size = self.in_channels
        uniform(size, self.root_param)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)
    
    def message(self, x_i, x_j, pseudo):
        weight_k = self.kernel(pseudo).view(-1, self.out_channels, self.out_channels)
        weight_op = self.operator_kernel(pseudo).view(-1, self.out_channels, self.out_channels)
        x_j_k = torch.matmul((x_j-x_i).unsqueeze(1), weight_k).squeeze(1)
        x_j_op = torch.matmul(x_j.unsqueeze(1), weight_op).squeeze(1)
        return x_j_k + x_j_op
    
    def update(self, aggr_out, x):
        return aggr_out + torch.mm(x, self.root_param) + self.bias
    
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
    