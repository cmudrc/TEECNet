import os
import torch
import numpy as np
from torch_geometric.data import Data
import matplotlib.pyplot as plt


def create_error_map(data_dir):
    # load data
    data = torch.load(data_dir)
    # get error map
    input_data = data.x
    label_data = data.y
    error_map = torch.abs(input_data - label_data)
    # create new data
    new_data = Data(x=error_map, y=data.y, edge_index=data.edge_index, edge_attr=data.edge_attr)
    return new_data

def compute_centralized_error(graph_data):
    # for each node, get the edges directly connected to it and get the edge length
    edge_index = graph_data.edge_index
    edge_attr = graph_data.edge_attr
    error_map = graph_data.x
    num_nodes = error_map.shape[0]
    avg_edge_length_list = []
    for i in range(num_nodes):
        # get the edges connected to node i
        edge_index_i = edge_index[:, edge_index[0, :] == i]
        edge_attr_i = edge_attr[edge_index[0, :] == i]
        # sum up the edge attributes connected to node i, and compute average edge length
        edge_attr_sum = torch.sum(edge_attr_i)
        avg_edge_length = edge_attr_sum / edge_attr_i.shape[0]
        avg_edge_length_list.append(avg_edge_length)

    # interpolate the relations between the error map and the edge length
    avg_edge_length_list = np.array(avg_edge_length_list)
    error_map = np.array(error_map)
    # plot the scatter plot of the error map and the edge length
    plt.scatter(avg_edge_length_list, error_map[:, 0])
    # fit a polynomial to the scatter plot
    z = np.polyfit(avg_edge_length_list, error_map[:, 0], 8)
    f = np.poly1d(z)
    # plot the fitted polynomial
    x_new = np.linspace(avg_edge_length_list.min(), avg_edge_length_list.max(), 50)
    y_new = f(x_new)
    plt.plot(x_new, y_new)

    return avg_edge_length_list, error_map

if __name__ == '__main__':
    data_dir = 'C:/research/data/processed'
    data_list = os.listdir(data_dir)

    # extract 1 sample as a test
    data = torch.load(os.path.join(data_dir, data_list[0]))
    # get error map
    error_map_data = create_error_map(os.path.join(data_dir, data_list[0]))
    # compute the error map
    avg_edge_length_list, error_map = compute_centralized_error(error_map_data)
    plt.show()