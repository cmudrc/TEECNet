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


def kmeans_torch(X, n_clusters, max_iter=300, tol=1e-4, device='cuda'):
    X = X.to(device)
    n_points = X.size(0)
    init_indices = torch.randint(0, n_points, (n_clusters,)).to(device)
    centroids = X[init_indices]
    
    for _ in range(max_iter):
        # Compute distances from points to centroids
        distances = torch.cdist(X, centroids)
        
        # Assign points to the nearest centroid
        assignments = torch.argmin(distances, dim=1)
        
        # Update centroids
        new_centroids = torch.stack([X[assignments == k].mean(dim=0) for k in range(n_clusters)], dim=0)
        
        # Check convergence
        if torch.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids
    
    return assignments


class MultiKernelConvGlobalAlphaWithEdgeConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, num_kernels, num_powers=4):
        super(MultiKernelConvGlobalAlphaWithEdgeConv, self).__init__(aggr='add')
        self.convs = nn.ModuleList()
        for i in range(num_powers):
            self.convs.append(pyg_nn.Linear(in_channels, out_channels, bias=False))
        # self.lin_similar = nn.Linear(in_channels+2, out_channels)
        self.lin = pyg_nn.Linear(in_channels, out_channels)

        self.alpha = nn.Parameter(torch.randn(num_kernels, num_powers, out_channels, out_channels))
        # self.parameter_activation = nn.Softplus()
        # self.coefficient = nn.Parameter(torch.full((num_kernels,), 1.0))
        self.n_powers = num_powers
        self.n_kernels = num_kernels
        
        self.activation = nn.LeakyReLU(0.1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, edge_index, edge_attr, cluster_assignments):
        # Apply alpha to each group and compute edge weights
        # x = self.lin(x)
        edge_weights_list = []
        edge_mask_list = []
        for k in range(self.n_kernels):
            node_mask = cluster_assignments == k
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            masked_edge_attr = edge_attr * edge_mask.float().unsqueeze(-1)

            for i in range(self.n_powers):
                edge_attr_power = self.convs[i](masked_edge_attr)
                if i == 0:
                    edge_attr_power_full = torch.mm(edge_attr_power, self.alpha[k, i].T)
                else:
                    edge_attr_power = self.activation(edge_attr_power)
                    edge_attr_power = torch.mm(torch.pow(edge_attr_power, i), self.alpha[k, i].T)
                    edge_attr_power_full = edge_attr_power_full + edge_attr_power

            # retrieve edge attributes for edges that belong to the current cluster
            # masked_edge_attr = edge_attr * edge_mask.float()
            # edge_weights = torch.pow(masked_edge_attr, self.alpha[k]) # this implementation goes to nan
            
            edge_weights_list.append(edge_attr_power_full)
            edge_mask_list.append(edge_mask)

        # Rearrange edge weights into its original position 
        combined_edge_weights = torch.zeros_like(edge_weights_list[0], device=self.device)
        for edge_mask_batch, edge_weights_batch in zip(edge_mask_list, edge_weights_list):
            combined_edge_weights[edge_mask_batch] = edge_weights_batch[edge_mask_batch]
        
        # Count the number of incoming edges for each node
        # num_edges_per_node = torch.bincount(edge_index[0], minlength=x.size(0)).float().to('cuda')
        num_edges_per_node = torch.zeros(x.size(0), device=self.device).scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0], device=self.device).float())
        
        # Normalize the combined edge weights by making sure that the edge weights for each node sum to 1
        normalized_edge_weights = combined_edge_weights / num_edges_per_node[edge_index[0]].view(-1, 1)
        # norm = torch.zeros(x.size(0), device=self.device).scatter_add_(0, edge_index[0], normalized_edge_weights)
        # norm_inv = torch.ones_like(norm) / norm

        # Compute the messages
        # msg = x[edge_index[1]] * normalized_edge_weights
        msg = normalized_edge_weights
        
        # Aggregate messages
        out = self.propagate(edge_index, x=msg)
        # self.raw_alpha = alpha
        # self.raw_coefficient = coefficient
        return out, normalized_edge_weights

    def message(self, x):
        return x
    

class KernelConv(pyg_nn.MessagePassing):
    def __init__(self, in_channel, out_channel, kernel, num_layers=1, **kwargs):
        super(KernelConv, self).__init__(aggr='add')
        self.in_channels = in_channel
        self.out_channels = out_channel
        self.root_param = nn.Parameter(torch.Tensor(in_channel, out_channel))
        self.bias = nn.Parameter(torch.Tensor(out_channel))

        self.kernel = kernel(in_channel=in_channel, out_channel=out_channel, num_layers=num_layers, **kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.kernel)
        size = self.in_channels
        uniform(size, self.root_param)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)
    
    def message(self, x_j, pseudo):
        weight = self.kernel(pseudo)
        x_j = x_j * weight
        return x_j
    
    def update(self, aggr_out, x):
        return aggr_out + self.bias + torch.mm(x, self.root_param)
    
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
    

class PowerSeriesConv(nn.Module):
    def __init__(self, in_channel, out_channel, num_powers, **kwargs):
        super(PowerSeriesConv, self).__init__()
        self.num_powers = num_powers
        self.convs = torch.nn.ModuleList()
        for i in range(num_powers):
            self.convs.append(nn.Linear(in_channel, out_channel))
        self.activation = nn.ReLU()
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
        self.conv_out = PowerSeriesConv(64, kwargs['out_channel'], num_powers)
        self.activation = activation()

    def forward(self, edge_attr):
        x = self.activation(self.conv0(edge_attr))
        for i in range(self.num_layers):
            x = self.activation(self.convs[i](x))
        x = self.conv_out(x)
        return x


class EllipseAreaNetwork(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels):
        super(EllipseAreaNetwork, self).__init__()
        # self.conv1 = MultiKernelConvGlobalAlphaWithEdgeConv(in_channels, 64, num_kernels, num_powers=3)
        # self.conv2 = MultiKernelConvGlobalAlphaWithEdgeConv(64, out_channels, num_kernels, num_powers=3)
        self.conv1 = MultiKernelConvGlobalAlphaWithEdgeConv(in_channels, out_channels, num_kernels, num_powers=3)
        self.lin_similar = nn.Linear(in_channels+2, out_channels)
        self.edge_conv = EdgeConv(nn.Sequential(
            nn.Linear(out_channels * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ), aggr='max')
        self.fc = torch.nn.Linear(1, 1)
        self.num_kernels = num_kernels
        self.alpha = None
        self.cluster = None
        # self.coefficient = None

    def forward(self, data):
        x, pos, edge_index, edge_attr = data.x, data.x, data.edge_index, data.edge_attr

        x_similar = self.lin_similar(torch.cat([x, pos], dim=1))
        x_similar = F.relu(x_similar)
        x_similar = self.edge_conv(x_similar, edge_index)
        x_similar = F.relu(x_similar)
        cluster_assignments = kmeans_torch(x_similar, self.num_kernels, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.cluster = cluster_assignments
        x, alpha1 = self.conv1(x, edge_index, edge_attr, cluster_assignments)
        # x = F.relu(x)
        # x, alpha2 = self.conv2(x, edge_index, edge_attr, cluster_assignments)
        x = F.relu(x)
        x = pyg_nn.pool.global_mean_pool(x, data.batch)
        # alpha = [alpha1, alpha2]
        # self.coefficient = [coefficient1, coefficient2]
        self.alpha = [alpha1]
        return self.fc(x)
    

class HeatTransferNetworkInterpolate(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_kernels, dropout=0.0):
        super(HeatTransferNetworkInterpolate, self).__init__()
        self.lin_similar = nn.Linear(in_channels+2, hidden_channels)
        self.edge_conv = EdgeConv(nn.Sequential(
            nn.Linear(hidden_channels * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ), aggr='max')
        self.conv1 = MultiKernelConvGlobalAlphaWithEdgeConv(in_channels, hidden_channels, num_kernels)
        self.act = torch.nn.LeakyReLU(0.1)
        self.conv2 = MultiKernelConvGlobalAlphaWithEdgeConv(hidden_channels, hidden_channels, num_kernels)
        self.conv4 = MultiKernelConvGlobalAlphaWithEdgeConv(hidden_channels, hidden_channels, num_kernels)
        # self.conv5 = MultiKernelConvGlobalAlphaWithEdgeConv(hidden_channels, out_channels, num_kernels)
        self.conv3 = pyg_nn.Linear(1 + hidden_channels, out_channels)
        self.dropout = dropout
        self.interpolate = graph_rdf_interpolation
        self.num_kernels = num_kernels
        self.alpha = None
        self.cluster = None
        # self.coefficient = None
        self.errors = None

    def forward(self, data):
        x, edge_index, edge_attr, pos, edge_index_high, edge_attr_high, pos_high = data.x, data.edge_index, data.edge_attr, data.pos, data.edge_index_high, data.edge_attr_high, data.pos_high
        # clusters = []
        alphas = []
        # coefficients = []
        errors = []
        x_similar = self.lin_similar(torch.cat([x, pos], dim=1))
        x_similar = F.relu(x_similar)
        x_similar = self.edge_conv(x_similar, edge_index)
        x_similar = F.relu(x_similar)
        cluster_assignments = kmeans_torch(x_similar, self.num_kernels, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.cluster = cluster_assignments
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        e, alpha = self.conv1(x, edge_index, edge_attr, cluster_assignments)
        alphas.append(alpha)
        
        errors.append(e)
       
        e, alpha = self.conv4(e, edge_index, edge_attr, cluster_assignments)
        alphas.append(alpha)

        e = self.interpolate(e, pos, pos_high, k=4)
        x = self.interpolate(x, pos, pos_high, k=4)

        e = self.conv3(torch.cat([e, x], dim=1))
        
        self.alpha = alphas
        # self.cluster = clusters
        # self.coefficient = coefficients
        self.errors = errors
        return e
    

class HeatTransferNetwork(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_kernels, dropout=0.0):
        super(HeatTransferNetwork, self).__init__()
        self.lin_similar = nn.Linear(in_channels+2, hidden_channels)
        self.lin_x = nn.Linear(in_channels, hidden_channels)
        self.lin_edge = nn.Linear(2, hidden_channels)

        self.edge_conv = EdgeConv(nn.Sequential(
            nn.Linear(hidden_channels * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ), aggr='max')
        self.conv1 = MultiKernelConvGlobalAlphaWithEdgeConv(hidden_channels, hidden_channels, num_kernels)
        self.act = torch.nn.LeakyReLU(0.1)
        self.conv2 = MultiKernelConvGlobalAlphaWithEdgeConv(hidden_channels, hidden_channels, num_kernels)
        self.conv4 = MultiKernelConvGlobalAlphaWithEdgeConv(hidden_channels, hidden_channels, num_kernels)
        self.conv5 = MultiKernelConvGlobalAlphaWithEdgeConv(hidden_channels, out_channels, num_kernels)
        self.neural_operator = KernelNN(width=41, ker_width=512, depth=6, ker_in=64, in_width=hidden_channels)
        # self.conv3 = pyg_nn.Linear(1 + hidden_channels, out_channels)
        self.dropout = dropout
        self.num_kernels = num_kernels
        self.alpha = None
        self.cluster = None
        # self.coefficient = None
        self.errors = None

    def forward(self, data):
        x, edge_index, pos = data.x, data.edge_index, data.pos
        # clusters = []
        x_lin = self.lin_x(x)
        alphas = []
        # coefficients = []
        errors = []
        x_similar = self.lin_similar(torch.cat([x, pos], dim=1))
        x_similar = F.relu(x_similar)
        x_similar = self.edge_conv(x_similar, edge_index)
        x_similar = F.relu(x_similar)
        cluster_assignments = kmeans_torch(x_similar, self.num_kernels, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.cluster = cluster_assignments
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        edge_attr = pos[edge_index[1]] - pos[edge_index[0]]
        edge_attr = self.lin_edge(edge_attr)

        e, edge_attr = self.conv1(x_lin, edge_index, edge_attr, cluster_assignments)
        # alphas.append(alpha)
        errors.append(e)


        e, edge_attr = self.conv2(e, edge_index, edge_attr, cluster_assignments)
        # alphas.append(alpha)

        e, edge_attr = self.conv4(e, edge_index, edge_attr, cluster_assignments)
        # alphas.append(alpha)

        e, edge_attr = self.conv5(e+x_lin, edge_index, edge_attr, cluster_assignments)
        # construct new Data object to feed into neural operator
        # x = self.lin_x(x)
        # e = self.neural_operator(torch.cat([x_lin, e], dim=1), edge_index, edge_attr)
        # e = self.act(e)
        # e, alpha = self.conv5(torch.cat([e, x], dim=1), edge_index_high, edge_attr_high, cluster_assignments)
        # alphas.append(alpha)

        self.alpha = alphas

        return e
    

class BurgerNetwork(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_kernels, dropout=0.0):
        super(BurgerNetwork, self).__init__()
        self.lin_similar = nn.Linear(in_channels+2, hidden_channels)
        self.lin_x = nn.Linear(in_channels+2, hidden_channels)
        self.lin_edge = nn.Linear(2, hidden_channels)

        self.edge_conv = EdgeConv(nn.Sequential(
            nn.Linear(hidden_channels * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ), aggr='max')
        self.act = torch.nn.LeakyReLU(0.1)
        
        # construct conv block with MultiKernelConvGlobalAlphaWithEdgeConv layer, actiavtion, and max pooling
        self.conv_block = nn.ModuleList()
        self.conv_block.append(MultiKernelConvGlobalAlphaWithEdgeConv(hidden_channels, hidden_channels, num_kernels))
        self.conv_block.append(nn.LeakyReLU(0.1))
        # self.conv_block.append(nn.MaxPool1d(3, stride=1, padding=1))

        # construct conv sequence with conv block
        self.convs = nn.ModuleList()
        for i in range(6):
            self.convs.extend(self.conv_block)

        self.neural_operator = KernelNN(width=41, ker_width=64, depth=6, ker_in=2, in_width=2*hidden_channels)
        # self.conv3 = pyg_nn.Linear(2*hidden_channels, out_channels)
        # self.conv3 = EncoderDecoder(2*hidden_channels, hidden_channels, out_channels)
        
        self.dropout = dropout
        self.num_kernels = num_kernels
        # self.alpha = None
        self.cluster = None
        # self.coefficient = None
        # self.errors = None

    def forward(self, data):
        x, edge_index, pos = data.x, data.edge_index, data.pos
        # clusters = []
        x_lin = self.lin_x(torch.cat([x, pos], dim=1))
        edge_attr = pos[edge_index[1]] - pos[edge_index[0]]
        edge_attr_lin= self.lin_edge(edge_attr)
        e = x_lin
        # alphas = []
        # coefficients = []
        # errors = []
        x_similar = self.lin_similar(torch.cat([x, pos], dim=1))
        x_similar = F.relu(x_similar)
        x_similar = self.edge_conv(x_similar, edge_index)
        x_similar = F.relu(x_similar)
        cluster_assignments = kmeans_torch(x_similar, self.num_kernels, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.cluster = cluster_assignments
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # run conv sequence
        for conv in self.convs:
            if isinstance(conv, MultiKernelConvGlobalAlphaWithEdgeConv):
                e, edge_attr_lin = conv(e, edge_index, edge_attr_lin, cluster_assignments)
                # errors.append(e)
            else:
                e = conv(e)

        # construct new Data object to feed into neural operator
        # x = self.lin_x(x)
        # e = self.conv3(torch.cat([e, x_lin], dim=1), edge_index)
        e = self.neural_operator(e, edge_index, edge_attr)
        # e = self.act(e)
        # e, alpha = self.conv5(torch.cat([e, x], dim=1), edge_index_high, edge_attr_high, cluster_assignments)
        # alphas.append(alpha)

        # self.alpha = alphas

        return e


def graph_rdf_interpolation(x, pos, pos_high, k=4):
    # x: [N, C]
    # pos: [N, 2]
    # pos_high: [M, 2]
    # k: int
    # return: [M, C]
    with torch.no_grad():
        pos = pos.unsqueeze(0).expand(pos_high.size(0), -1, -1)  # [M, N, 2]
        pos_high = pos_high.unsqueeze(1).expand(-1, pos.size(1), -1)  # [M, N, 2]
        dist = torch.norm(pos - pos_high, dim=-1)  # [M, N]
        _, indices = torch.topk(dist, k, dim=-1, largest=False)  # [M, k]
        x = x.unsqueeze(0).expand(pos_high.size(0), -1, -1)  # [M, N, C]
        x = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))  # [M, k, C]
        x = x.mean(dim=1)  # [M, C]
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
    

class EncoderDecoderOld(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(EncoderDecoderOld, self).__init__()
        self.encoder = EncoderOld(in_channels, hidden_channels)
        self.message_passing = pyg_nn.GCNConv(hidden_channels, hidden_channels)
        self.decoder = DecoderOld(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x1, x2 = self.encoder(x, edge_index)
        x = self.message_passing(x2, edge_index)
        dec_input = x1 + x
        dec_output = self.decoder(dec_input, edge_index)
        return dec_output
    

class EncoderOld(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(EncoderOld, self).__init__()
        self.layer1 = ConvResidualBlock(in_channels, hidden_channels)
        self.layer2 = ConvResidualBlock(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x1 = self.layer1(x, edge_index)
        x2 = self.layer2(x1, edge_index)
        return x1, x2
    

class DecoderOld(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(DecoderOld, self).__init__()
        self.layer1 = ConvResidualBlock(hidden_channels, hidden_channels)
        self.layer2 = ConvResidualBlock(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x1 = self.layer1(x, edge_index)
        x2 = self.layer2(x1, edge_index)
        return x2
    

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
    

class CFDErrorInterpolateOld(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CFDErrorInterpolateOld, self).__init__()
        self.error = CFDError(2, 3)
        self.combine = EncoderDecoderOld(6, 64, 3)
        self.interpolate = ErrorInterpolate()

    def forward(self, data_l, data_h):
        x, edge_index, pos_l = data_l.x, data_h.edge_index, data_l.pos
        pos_h = data_h.pos

        e = self.error(data_l)
        x = torch.cat([x, e], dim=1)
        x = self.interpolate(x, pos_l, pos_h)
        x = self.combine(x, edge_index) 
        return x