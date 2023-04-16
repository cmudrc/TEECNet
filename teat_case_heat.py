import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import knn_interpolate
# import torch.nn.functional as F
# from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from dataset.MatDataset import MatDataset
from model.cfd_error import MultiKernelConvGlobalAlphaWithEdgeConv
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
from utils import train_test_split


class HeatTransferDataset(MatDataset):
    def __init__(self, root, transform=None, pre_transform=None, res_low=0, res_high=3):
        super(HeatTransferDataset, self).__init__(root, transform, pre_transform)
        self.res_low = res_low
        self.res_high = res_high
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)
    
    @property
    def raw_file_names(self):
        return ['heat_solutions_res_10.h5', 'heat_solutions_res_20.h5', 'heat_solutions_res_40.h5', 'heat_solutions_res_80.h5']
    
    @property
    def mesh_file_names(self):
        return ['mesh_res_10.h5', 'mesh_res_20.h5', 'mesh_res_40.h5', 'mesh_res_80.h5']

    @property
    def processed_file_names(self):
        return ['heat_transfer_data.pt']

    def process(self):
        data_list = []
        mesh_resolutions = [self.res_low, self.res_high]
        # load mesh
        X_list = []
        lines_list = []
        lines_length_list = []
        for res in mesh_resolutions:
            with h5py.File(os.path.join(self.raw_dir, self.processed_file_names[res]), 'r') as f:
                X = f['X']
                lines = f['lines']
                lines_length = f['lines_lengths']
                X_list.append(X)
                lines_list.append(lines)
                lines_length_list.append(lines_length)
        
        for i in range(500):
            x_all = []
            edge_index_all = []
            edge_attr_all = []
            pos_all = []
            for res in mesh_resolutions:
                with h5py.File(os.path.join(self.raw_dir, self.raw_file_names[res]), 'r') as f:
                    data_array = f['u_sim_{}'.format(i)][:]
                    x = torch.tensor(data_array, dtype=torch.float)
                    x_all.append(x)
                    edge_index = torch.tensor(lines_list[mesh_resolutions.index(int(res))], dtype=torch.int32)
                    edge_index_all.append(edge_index)
                    edge_attr = torch.tensor(lines_length_list[mesh_resolutions.index(int(res))], dtype=torch.float)
                    edge_attr_all.append(edge_attr)
                    pos = torch.tensor(X_list[mesh_resolutions.index(int(res))], dtype=torch.float)
                    pos_all.append(pos)
                    
            data = Data(x=x_all[0], edge_index=edge_index_all[0], edge_attr=edge_attr_all[0], pos=pos_all[0], edge_index_high=edge_index_all[1], edge_attr_high=edge_attr_all[1], pos_high=pos_all[1], y=x_all[1])
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class HeatTransferNetwork(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_kernels, dropout=0.0):
        super(HeatTransferNetwork, self).__init__()
        self.conv1 = MultiKernelConvGlobalAlphaWithEdgeConv(in_channels, hidden_channels, num_kernels)
        self.conv2 = MultiKernelConvGlobalAlphaWithEdgeConv(hidden_channels, hidden_channels, num_kernels)
        self.conv3 = MultiKernelConvGlobalAlphaWithEdgeConv(hidden_channels, out_channels, num_kernels)
        self.dropout = dropout
        self.interpolate = knn_interpolate

    def forward(self, data):
        x, edge_index, edge_attr, pos, edge_index_high, edge_attr_high, pos_high = data.x, data.edge_index, data.edge_attr, data.pos, data.edge_index_high, data.edge_attr_high, data.pos_high
        e = self.conv1(x, edge_index, edge_attr)
        e = self.conv2(x, edge_index, edge_attr)
        e = self.interpolate(e, pos, pos_high)
        x = self.interpolate(x, pos, pos_high)
        x = x + e
        return x
    
def visualize_alpha(writer, model, epoch):
    alphas = model.alpha[1]
    # alphas = np.array(alphas, dtype=np.float32)
    writer.add_histogram("Alpha", alphas, epoch)

def visualize_clusters(writer, data, model, epoch):
    clusters = model.cluster[1]
    # clusters = clusters.detach().cpu().numpy()

    fig = plt.figure()
    plt.scatter(data.x[:, 0].detach().cpu().numpy(), data.x[:, 1].detach().cpu().numpy(), c=clusters.detach().cpu().numpy(), cmap="viridis")
    plt.colorbar()
    plt.title(f"Clusters (Epoch: {epoch})")

    writer.add_figure("Clusters", fig, epoch)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeatTransferNetwork(1, 64, 1, 3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    dataset = HeatTransferDataset('dataset/heat', res_low=0, res_high=3)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    writer = SummaryWriter('runs/heat_transfer')
    for epoch in range(500):
        model.train()
        loss_all = 0
        for data in tqdm(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = torch.nn.functional.mse_loss(out, data.y)
            # r2_accuracy = r2_score(data.y.cpu().detach().numpy(), out.cpu().detach().numpy())
            loss.backward()
            loss_all += loss.item()
            optimizer.step()
        writer.add_scalar('Loss/train', loss_all / len(train_loader), epoch)
        visualize_alpha(writer, model, epoch)
        visualize_clusters(writer, data, model, epoch)
        print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss_all / len(train_loader)))

        if epoch % 10 == 0:
            model.eval()
            loss_all = 0
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                loss = torch.nn.functional.mse_loss(out, data.y)
                loss_all += loss.item()
            writer.add_scalar('Loss/test', loss_all / len(test_loader), epoch)
            print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss_all / len(test_loader)))
    torch.save(model.state_dict(), 'test_cases/heat/model.pt')
    writer.close()
