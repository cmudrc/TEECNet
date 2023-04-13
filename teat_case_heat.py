import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch_geometric.data import Data
# import torch.nn.functional as F
# from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from dataset.MatDataset import MatDataset
from model.cfd_error import EllipseAreaNetwork
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py


class HeatTransferDataset(MatDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(HeatTransferDataset, self).__init__(root, transform, pre_transform)
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
        mesh_resolutions = [10, 20, 40, 80]
        # load mesh
        X_list = []
        lines_list = []
        lines_length_list = []
        for file in self.mesh_file_names:
            with h5py.File(os.path.join(self.raw_dir, file), 'r') as f:
                X = f['X']
                lines = f['lines']
                lines_length = f['lines_lengths']
                X_list.append(X)
                lines_list.append(lines)
                lines_length_list.append(lines_length)

        for file in self.raw_file_names:
            str1 = file.split('_')[3]
            res = str1.split('.')[0]
            with h5py.File(os.path.join(self.raw_dir, file), 'r') as f:
                for i in range(500):
                    data_array = f['u_sim_{}'.format(i)][:]
                    x = torch.tensor(data_array, dtype=torch.float)
                    edge_index = torch.tensor(lines_list[mesh_resolutions.index(int(res))], dtype=torch.int32)
                    edge_attr = torch.tensor(lines_length_list[mesh_resolutions.index(int(res))], dtype=torch.float)
                    pos = torch.tensor(X_list[mesh_resolutions.index(int(res))], dtype=torch.float)
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

                    data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
