import os
import yaml
from argparse import ArgumentParser
from datetime import datetime
import torch
import torch_geometric.nn as pyg_nn
from model.cfd_error import EllipseAreaNetwork, HeatTransferNetwork
from model.neural_operator import KernelNN
from model.GraphSAGE import GraphSAGE
# from dataset.MegaFlow2D import MegaFlow2D

from megaflow.dataset.MegaFlow2D import MegaFlow2D
from dataset.MatDataset import HeatTransferDataset
from metrics.metrics_all import *
from torch_geometric.data import Batch
import meshio
import numpy as np


def collate_fn(data_list):
    data_list_l, data_list_h = zip(*data_list)
    batched_data_l = Batch.from_data_list(data_list_l)
    batched_data_h = Batch.from_data_list(data_list_h)
    return batched_data_l, batched_data_h

def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def checkpoint_save(model, name, epoch):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch


def initialize_model(type, in_channel, out_channel, *args, **kwargs):
    # initialize model based on type, layers, and num_filters provided
    if type == 'GraphSAGE':
        model = pyg_nn.GraphSAGE(in_channel, kwargs['hidden_channel'], kwargs['num_layers'], out_channel, kwargs['dropout'])
    # elif type == 'CFDError':
    #     model = CFDError(in_channel, out_channel)
    # elif type == 'CFDErrorInterpolate':
    #     model = CFDErrorInterpolate(in_channel, out_channel)
    # elif type == 'CFDErrorInterpolateOld':
    #     model = CFDErrorInterpolateOld(in_channel, out_channel)
    elif type == 'EllipseArealNetwork':
        model = EllipseAreaNetwork(in_channel, out_channel, kwargs['num_kernels'])
    elif type == 'HeatTransferNetwork':
        model = HeatTransferNetwork(in_channel, kwargs['hidden_channel'], out_channel, kwargs['num_kernels'])
    elif type == 'NeuralOperator':
        model = KernelNN(kwargs['width'], kwargs['ker_width'], kwargs['depth'], in_channel, out_channel)
    else:
        raise ValueError('Unknown model type: {}'.format(type))
    return model


def initialize_dataset(dataset, **kwargs):
    # initialize dataset based on dataset and mode
    if dataset == 'MegaFlow2D':
        dataset = MegaFlow2D(root=dir, download=False, split_scheme=kwargs['split_scheme'], transform=kwargs['transform'], pre_transform=kwargs['pre_transform'], split_ratio=kwargs['split_ratio'])
        print('Dataset initialized')
    elif dataset == 'HeatTransferDataset':
        dataset = HeatTransferDataset(root='dataset/heat', res_low=kwargs['res_low'], res_high=kwargs['res_high'])
        print('Dataset initialized')
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
    return dataset


def initialize_loss(loss_type):
    """
    Initialize loss function based on type provided
    Input:
        loss_type: string, type of loss function
    Output:
        loss_fn: loss function
    """
    if loss_type == 'MSELoss':
        return MSE()
    elif loss_type == 'L1Loss':
        return L1()
    elif loss_type == 'VorticityLoss':
        return VorticityLoss()
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))


def initialize_metric(metric_type):
    """
    Initialize metric function based on type provided
    Input:
        metric_type: string, type of metric function
    Output:
        metric_fn: metric function
    """
    if metric_type == 'max_divergence':
        return Accuracy(name='max_divergence', prediction_fn=None)
    elif metric_type == 'norm_divergence':
        return Accuracy(name='norm_divergence', prediction_fn=None)
    elif metric_type == 'r2_score':
        return Accuracy(name='r2_score', prediction_fn=None)
    else:
        raise ValueError('Unknown metric type: {}'.format(metric_type))


def evaluate_model(model, dataloader, logger, iteration, loss_fn, eval_metric, device, mode, checkpoint=None):
    # load checkpoint if provided
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        avg_loss = 0
        avg_metric = 0

        for (batch_l, batch_h) in dataloader:
            batch_l, batch_h = batch_l.to(device), batch_h.to(device)
            pred = model(batch_l, batch_h)
            loss = loss_fn.compute(batch_h.x, pred, batch_h.pos, batch_h.edge_index, weight=0.001)
            avg_loss += loss.item()
            avg_metric += eval_metric.compute(batch_h.x, pred).item()

        avg_loss /= len(dataloader)
        avg_metric /= len(dataloader)

        if mode == 'val':
            logger.add_scalar('Loss/val', avg_loss, iteration)
            logger.add_scalar('Max_div/val', avg_metric, iteration)
            print('-' * 72)
            print('Val loss: {:.4f}, Val metric: {:.4f}'.format(avg_loss, avg_metric))

        if mode == 'test':
            logger.add_scalar('test_loss', avg_loss, iteration)
            logger.add_scalar('test_metric', avg_metric, iteration)
            print('-' * 72)
            print('Test loss: {:.4f}, Test metric: {:.4f}'.format(avg_loss, avg_metric))

    model.train()
    return avg_loss, avg_metric


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MegaFlow2D', help='dataset name')
    parser.add_argument('--split_scheme', type=str, default='mixed', help='dataset mode')
    parser.add_argument('--transform', type=str, default='None', help='dataset transform')
    parser.add_argument('--dir', type=str, default='C:/research/data', help='dataset directory')
    parser.add_argument('--model', type=str, default='FlowMLConvolution', help='model type')
    parser.add_argument('--layers', type=int, default=3, help='number of layers')
    parser.add_argument('--num_filters', type=int, nargs='+', default=[8, 16, 8], help='number of filters')
    parser.add_argument('--loss', type=str, default='MSELoss', help='loss function')
    parser.add_argument('--metric', type=str, default='max_divergence', help='metric function')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--load_model', type=str, default=None, help='load model from checkpoint')

    args = parser.parse_args()
    return args

def load_yaml(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def save_yaml(config, path):
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def train_test_split(dataset, train_ratio):
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]
    
    return train_dataset, test_dataset
