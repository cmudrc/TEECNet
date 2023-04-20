import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import knn_interpolate
# import torch.nn.functional as F
# from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from dataset.MatDataset import MatDataset
from model.cfd_error import MultiKernelConvGlobalAlphaWithEdgeConv
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
from test_case_heat import HeatTransferDataset
from test_case_heat import HeatTransferNetwork


def plot_prediction(model, dataset, res_low, res_high, idx):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[idx]

    data = data.to(device)

    # send to model and get prediction
    pred = model(data)
    X = data.pos_high[:, 0].reshape(res_high, res_high).cpu().numpy()
    Y = data.pos_high[:, 1].reshape(res_high, res_high).cpu().numpy()
    # plot prediction as a contour plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].contourf(X, Y, data.y.reshape(res_high, res_high).detach().cpu().numpy())
    ax[0].set_title('Ground Truth')
    ax[1].contourf(X, Y, pred.reshape(res_high, res_high).detach().cpu().numpy())
    ax[1].set_title('Prediction')
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up dataset
    root = 'dataset/heat'
    dataset = HeatTransferDataset(root, res_low=0, res_high=1)
    print(dataset[480])
    # set up model
    model = HeatTransferNetwork(1, 64, 1, 3).to(device)
    # load model
    model.load_state_dict(torch.load('test_cases/heat_transfer/2023-04-18_11-26/model_470.pt'))
    # plot prediction
    plot_prediction(model, dataset, 11, 21, 0)
    
if __name__ == '__main__':
    main()