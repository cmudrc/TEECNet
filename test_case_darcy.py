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


class DarcyDataset(MatDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DarcyDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['darcy_data.mat']

    @property
    def processed_file_names(self):
        return ['darcy_data.pt']

    def process(self):
        data_list = []
        for i in range(len(self.raw_data['p'])):
            x = torch.tensor(self.raw_data['p'][i], dtype=torch.float)
            y = torch.tensor(self.raw_data['area'][i], dtype=torch.float)
            data = Data(x=x, y=y)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])