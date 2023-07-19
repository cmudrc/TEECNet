import os
import torch
import numpy as np
import scipy.io
import h5py
import sklearn.metrics
from torch_geometric.data import Data, InMemoryDataset
import torch.nn as nn
from scipy.ndimage import gaussian_filter


# function for debug purpose
def print_groups_and_datasets(name, obj):
    print(name, ":", type(obj))
    

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class MatDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MatDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        pass

    def process(self):
        raise NotImplementedError
    
    def extract_solution(self, h5_file, sim, res):
        raise NotImplementedError
    
    def construct_data_object(self, coords, connectivity, solution, k):
        raise NotImplementedError


class HeatTransferDataset(MatDataset):
    def __init__(self, root, transform=None, pre_transform=None, res_low=1, res_high=3):
        self.res_low = res_low
        self.res_high = res_high
        self.pre_transform = pre_transform
        # self.res_list = [10, 20, 40, 80]
        super(HeatTransferDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)
    
    @property
    def raw_file_names(self):
        return ['heat_solutions_res_8.h5', 'heat_solutions_res_16.h5', 'heat_solutions_res_32.h5', 'heat_solutions_res_64.h5']
    
    @property
    def mesh_file_names(self):
        return ['mesh_res_8.h5', 'mesh_res_16.h5', 'mesh_res_32.h5', 'mesh_res_64.h5']

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
            with h5py.File(os.path.join(self.raw_dir, self.mesh_file_names[res]), 'r') as f:
                X = f['X'][:]
                lines = f['lines'][:]
                lines_length = f['line_lengths'][:]
                X_list.append(X)
                lines_list.append(lines)
                lines_length_list.append(lines_length)

        for i in range(1000):
            x_all = []
            edge_index_all = []
            edge_attr_all = []
            pos_all = []
            for res in mesh_resolutions:
                with h5py.File(os.path.join(self.raw_dir, self.raw_file_names[res]), 'r') as f:
                    if self.pre_transform == 'interpolate_low':
                        # overide res to the lowest resolution
                        res = self.res_low
                    elif self.pre_transform == 'interpolate_high':
                        # overide res to the highest resolution
                        res = self.res_high
                    # for debug purpose list all the keys
                    # f.visititems(print_groups_and_datasets)
                    data_array = f['u_sim_{}'.format(i)][:]
                    x = torch.tensor(data_array, dtype=torch.float).unsqueeze(1)
                    x_all.append(x)
                    edge_index = torch.tensor(lines_list[mesh_resolutions.index(int(res))], dtype=torch.long).t().contiguous()
                    edge_index_all.append(edge_index)
                    edge_attr = torch.tensor(lines_length_list[mesh_resolutions.index(int(res))], dtype=torch.float)
                    edge_attr_all.append(edge_attr)
                    pos = torch.tensor(X_list[mesh_resolutions.index(int(res))], dtype=torch.float)
                    pos_all.append(pos)

            # normalize x and y to the scale of [0, 1]
            x_all[0] = (x_all[0] - x_all[0].min()) / (x_all[0].max() - x_all[0].min())
            # x_all[1] = (x_all[1] - x_all[1].min()) / (x_all[1].max() - x_all[1].min())
            x_all[1] = (x_all[1] - x_all[1].min()) / (x_all[1].max() - x_all[1].min())
            
            if self.pre_transform == 'interpolate_high':
                data = Data(x=x_all[0], edge_index=edge_index_all[1], edge_attr=edge_attr_all[1], pos=pos_all[1], edge_index_high=edge_index_all[1], edge_attr_high=edge_attr_all[1], pos_high=pos_all[1], y=x_all[1])
            else:
                data = Data(x=x_all[0], edge_index=edge_index_all[0], edge_attr=edge_attr_all[0], pos=pos_all[0], edge_index_high=edge_index_all[1], edge_attr_high=edge_attr_all[1], pos_high=pos_all[1], y=x_all[1])
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class BurgersDataset(MatDataset):
    def __init__(self, root, transform=None, pre_transform=None, res_low=1, res_high=3):
        self.res_low = res_low
        self.res_high = res_high
        self.pre_transform = pre_transform
        super(BurgersDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['solution_10.h5', 'solution_40.h5']
    
    @property
    def mesh_file_names(self):
        return ['mesh_10.h5', 'mesh_40.h5']
    
    @property
    def processed_file_names(self):
        return ['burgers_data.pt']
    
    def process(self):
        data_list = []
        mesh_resolutions = [self.res_low, self.res_high]
        # load mesh
        X_list = []
        lines_list = []
        lines_length_list = []
        for res in mesh_resolutions:
            with h5py.File(os.path.join(self.raw_dir, self.mesh_file_names[res]), 'r') as f:
                X = f['X'][:]
                lines = f['lines'][:]
                lines_length = f['line_lengths'][:]
                X_list.append(X)
                lines_list.append(lines)
                lines_length_list.append(lines_length)

        for i in range(1000):
            x_all = []
            edge_index_all = []
            edge_attr_all = []
            pos_all = []
            for res in mesh_resolutions:
                edge_index = torch.tensor(lines_list[mesh_resolutions.index(int(res))], dtype=torch.long).t().contiguous()
                edge_index_all.append(edge_index)
                edge_attr = torch.tensor(lines_length_list[mesh_resolutions.index(int(res))], dtype=torch.float).unsqueeze(1)
                edge_attr_all.append(edge_attr)
                pos = torch.tensor(X_list[mesh_resolutions.index(int(res))], dtype=torch.float)
                pos_all.append(pos)
        
                with h5py.File(os.path.join(self.raw_dir, self.raw_file_names[res]), 'r') as f:  
                    if self.pre_transform == 'interpolate_low':
                        # overide res to the lowest resolution
                        res = self.res_low
                    elif self.pre_transform == 'interpolate_high':
                        # overide res to the highest resolution
                        res = self.res_high
                    # for debug purpose list all the keys
                    # f.visititems(print_groups_and_datasets)
                    data_array_group = f['{}'.format(i)]
                    dset = data_array_group['u'][:]
                    # take one sample from each timeline as an example
                    x = torch.tensor(dset[90], dtype=torch.float).unsqueeze(1)
                    x_all.append(x)

            # normalize x and y to the scale of [0, 1]
            x_all[0] = (x_all[0] - x_all[0].min()) / (x_all[0].max() - x_all[0].min())
            x_all[1] = (x_all[1] - x_all[1].min()) / (x_all[1].max() - x_all[1].min())
            # x_all[1] = (x_all[1] - x_all[0].min()) / (x_all[1].max() - x_all[0].min())

            if self.pre_transform == 'interpolate_high':
                data = Data(x=x_all[0], edge_index=edge_index_all[1], edge_attr=edge_attr_all[1], pos=pos_all[1], edge_index_high=edge_index_all[1], edge_attr_high=edge_attr_all[1], pos_high=pos_all[1], y=x_all[1])
            else:
                data = Data(x=x_all[0], edge_index=edge_index_all[0], edge_attr=edge_attr_all[0], pos=pos_all[0], edge_index_high=edge_index_all[1], edge_attr_high=edge_attr_all[1], pos_high=pos_all[1], y=x_all[1])
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])