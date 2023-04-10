import os
import torch
import numpy as np
import scipy.io
import h5py
import sklearn.metrics
from torch_geometric.data import Data, InMemoryDataset
import torch.nn as nn
from scipy.ndimage import gaussian_filter


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
    def __init__(self, root, k=6, transform=None, pre_transform=None):
        self.k = k
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
    

class SteadyStateHeatDataset(MatDataset):
    def __init__(self, root, k=6, transform=None, pre_transform=None):
        super(SteadyStateHeatDataset, self).__init__(root, k, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['steady_state_simulations.h5']
    
    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        with h5py.File(os.path.join(self.raw_dir, self.raw_file_names[0]), 'r') as f:
            num_simulations = 500
            mesh_resolutions = [10, 20, 40, 80]
            for sim in range(num_simulations):
                for res in mesh_resolutions:
                    coords, connectivity, temperature = self.extract_solution(f, sim, res)
                    data = self.construct_data_object(coords, connectivity, temperature, k=self.k)
                    data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def extract_solution(self, h5_file, sim, res):
        group_name = f"u_sim_{sim}_res_{res}"
        mesh_coords = h5_file[f"{group_name}/Mesh/0/Geometry/coordinates"][:]
        mesh_connectivity = h5_file[f"{group_name}/Mesh/0/Topology/connectivity"][:]
        temperature = h5_file[f"{group_name}/Vector/0"][:]

        return mesh_coords, mesh_connectivity, temperature

    def construct_data_object(self, coords, connectivity, solution, k):
        x = torch.from_numpy(coords).float()
        edge_index = torch.from_numpy(connectivity).long()

        y = torch.from_numpy(solution).float()

        data = Data(x=x, edge_index=edge_index, y=y)
        return data