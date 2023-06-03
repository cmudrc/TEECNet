import torch
from torch_geometric.utils import add_self_loops, degree
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.nn import EdgeConv, knn_graph
import torch.nn.functional as F
from torch_scatter import scatter_softmax


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
            self.convs.append(pyg_nn.Linear(1, 1))
        # self.lin_similar = nn.Linear(in_channels+2, out_channels)
        self.lin = pyg_nn.Linear(in_channels, out_channels)

        self.alpha = nn.Parameter(torch.randn(num_kernels, num_powers, out_channels))
        # self.parameter_activation = nn.Softplus()
        # self.coefficient = nn.Parameter(torch.full((num_kernels,), 1.0))
        self.n_powers = num_powers
        self.n_kernels = num_kernels
        
        self.activation = nn.LeakyReLU(0.1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, edge_index, edge_attr, cluster_assignments):
        # Apply alpha to each group and compute edge weights
        x = self.lin(x)
        edge_weights_list = []
        edge_mask_list = []
        for k in range(self.n_kernels):
            node_mask = cluster_assignments == k
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            masked_edge_attr = edge_attr * edge_mask.float()

            for i in range(self.n_powers):
                edge_attr_power = self.convs[i](masked_edge_attr.unsqueeze(-1))
                edge_attr_power = self.activation(edge_attr_power)
                edge_attr_power = torch.pow(edge_attr_power, i+1) * self.alpha[k, i]
                if i == 0:
                    edge_attr_power_full = edge_attr_power
                else:
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
        
        # Normalize the combined edge weights
        # normalized_edge_weights = combined_edge_weights / num_edges_per_node[edge_index[0]] + 1e-8
        normalized_edge_weights = combined_edge_weights / num_edges_per_node[edge_index[0]].view(-1, 1) + 1e-8

        # Compute the messages
        # msg = x[edge_index[1]] * normalized_edge_weights
        msg = normalized_edge_weights
        # if torch.isnan(msg).any():
        #     print('nan in msg')
        #     exit()
        
        # Aggregate messages
        out = self.propagate(edge_index, x=msg)
        # if torch.isnan(out).any():
        #     print('nan in out')
        #     exit()
        # self.raw_alpha = alpha
        # self.raw_coefficient = coefficient
        return out, self.alpha

    def message(self, x):
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
        self.interpolate = graph_unpool
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


        e, alpha = self.conv2(e, edge_index, edge_attr, cluster_assignments)
        alphas.append(alpha)

        e, alpha = self.conv4(e, edge_index, edge_attr, cluster_assignments)
        alphas.append(alpha)

        e = self.conv3(torch.cat([e, x], dim=1))
        e = self.act(e)

        self.alpha = alphas

        return e



def graph_unpool(x_low, pos_low, pos_high, k=3, alpha=1.0, chunk_size=100):
    # Normalize low-resolution node features
    x_low_norm = x_low / x_low.norm(dim=-1, keepdim=True)

    x_high = torch.zeros(pos_high.size(0), x_low.size(1), device=x_low.device)

    # Loop over high-resolution nodes in chunks
    for i in range(0, pos_high.size(0), chunk_size):
        j = min(i + chunk_size, pos_high.size(0))

        # Calculate pairwise distances between nodes in current chunk and all low-resolution nodes
        dists = torch.cdist(pos_high[i:j], pos_low)

        # Find indices of k nearest neighbors
        _, idx = dists.topk(k, dim=1, largest=False)

        # Get k smallest distances
        dist = dists.gather(1, idx)

        # Compute radial basis function weights
        weights = torch.exp(-alpha * dist**2)

        # Normalize weights
        weights = weights / weights.sum(dim=1, keepdim=True)

        # Perform weighted sum to get the values at the high-resolution graph nodes
        x_high[i:j] = torch.bmm(weights.unsqueeze(1), x_low_norm[idx]).squeeze(1)

    return x_high


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