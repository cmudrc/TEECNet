import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch_geometric.data import Data
# import torch.nn.functional as F
# from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from model.cfd_error import EllipseAreaNetwork
import matplotlib.pyplot as plt


def load_ellipse_dataset(data_dir, hyperparams_dir):
    data = torch.load(data_dir)
    hyperparams = torch.load(hyperparams_dir)
    return data, hyperparams

def load_model(model_dir, num_kernels):
    model = EllipseAreaNetwork(num_kernels)
    model.load_state_dict(torch.load(model_dir))
    return model

def eval_performance_zero_shot_super_res(model, data, hyperparams):
    model.eval()

    mesh_res = [hyperparams[i][3] for i in range(len(hyperparams))]

    data_res_5 = [data[i] for i in range(len(data)) if mesh_res[i] == 0.5]
    data_res_6 = [data[i] for i in range(len(data)) if mesh_res[i] == 0.6]
    data_res_7 = [data[i] for i in range(len(data)) if mesh_res[i] == 0.7]
    data_res_8 = [data[i] for i in range(len(data)) if mesh_res[i] == 0.8]
    data_res_9 = [data[i] for i in range(len(data)) if mesh_res[i] == 0.9]

    data_res_5 = DataLoader(data_res_5, batch_size=1, shuffle=False)
    data_res_6 = DataLoader(data_res_6, batch_size=1, shuffle=False)
    data_res_7 = DataLoader(data_res_7, batch_size=1, shuffle=False)
    data_res_8 = DataLoader(data_res_8, batch_size=1, shuffle=False)
    data_res_9 = DataLoader(data_res_9, batch_size=1, shuffle=False)

    test_loss_list = []
    test_loss = 0
    with torch.no_grad():
        for data in data_res_5:
            data = data.to('cuda')
            out = model(data)
            loss = torch.nn.MSELoss()(out, data.y)
            test_loss += loss.item()
        test_loss = test_loss / len(data_res_5)
        test_loss_list.append(test_loss)
        
        for data in data_res_6:
            data = data.to('cuda')
            out = model(data)
            loss = torch.nn.MSELoss()(out, data.y)
            test_loss += loss.item()
        test_loss = test_loss / len(data_res_6)
        test_loss_list.append(test_loss)

        for data in data_res_7:
            data = data.to('cuda')
            out = model(data)
            loss = torch.nn.MSELoss()(out, data.y)
            test_loss += loss.item()
        test_loss = test_loss / len(data_res_7)
        test_loss_list.append(test_loss)

        for data in data_res_8:
            data = data.to('cuda')
            out = model(data)
            loss = torch.nn.MSELoss()(out, data.y)
            test_loss += loss.item()
        test_loss = test_loss / len(data_res_8)
        test_loss_list.append(test_loss)

        for data in data_res_9:
            data = data.to('cuda')
            out = model(data)
            loss = torch.nn.MSELoss()(out, data.y)
            test_loss += loss.item()
        test_loss = test_loss / len(data_res_9)
        test_loss_list.append(test_loss)

    return test_loss_list

def main():
    data_dir = 'test_cases\ellipse\dataset.pt'
    hyperparams_dir = 'test_cases\ellipse\hyperparameters.pt'
    model_dir = 'test_cases\ellipse\model.pt'
    num_kernels = 3
    data, hyperparams = load_ellipse_dataset(data_dir, hyperparams_dir)
    model = load_model(model_dir, num_kernels)
    model.to('cuda')
    test_loss_list = eval_performance_zero_shot_super_res(model, data, hyperparams)
    mesh_res = [0.5, 0.6, 0.7, 0.8, 0.9]
    plt.plot(mesh_res, test_loss_list)
    plt.xlabel('Mesh Resolution')
    plt.ylabel('MSE Loss')
    plt.savefig('test_cases\ellipse\performance.png')

if __name__ == '__main__':
    main()
