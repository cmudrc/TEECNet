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
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import h5py
from utils import initialize_model, initialize_dataset


def plot_prediction(model, dataset, res_low, res_high, idx):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[idx]

    data = data.to(device)

    # send to model and get prediction
    pred = model(data)
    X = data.pos_high[:, 0].cpu().numpy()
    Y = data.pos_high[:, 1].cpu().numpy()
    # reconstruct triangular element via edge_index
    tri_idx = data.edge_index_high.cpu().numpy().T
    tri = tri.Triangulation(X, Y, triangles=tri_idx)
    # plot prediction
    fig = plt.figure()
    plt.tricontourf(tri, pred.detach().cpu().numpy().flatten())
    plt.colorbar()
    plt.title('Prediction')
    
    # save figure
    plt.savefig('test_cases/heat_transfer/2023-04-25_14-02/prediction_{}.png'.format(idx))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up dataset
    root = 'dataset/heat'
    dataset = initialize_dataset(root, res_low=1, res_high=3)
    print(dataset[480])
    # set up model
    model = initialize_model(in_channel=1, hidden_channel=64, out_channel=1, num_kernels=2)
    # load model
    model.load_state_dict(torch.load('test_cases/heat_transfer/2023-04-25_14-02/model_50.pt'))
    # plot prediction
    plot_prediction(model, dataset, 11, 41, 0)
    
if __name__ == '__main__':
    main()