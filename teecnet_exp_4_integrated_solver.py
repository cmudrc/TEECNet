import os
import time
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
# from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
from utils import train_test_split, get_cur_time, parse_args, load_yaml
from model.solver import IntergratedBurgersSolver, BurgersSolver

from fenics import *
from dolfin import *


def gen_random_expression_str_2d():
    """
    generate a str expression for initial condition of burgers equation using a Gaussian initial velocity distribution. The center of the Gaussian is randomly generated.
    """
    x_center = np.random.uniform(0, 3)
    y_center = np.random.uniform(0, 3)
    return 'exp(-2*pow(x[0] - ' + str(x_center) + ', 2) - 2*pow(x[1] - ' + str(y_center) + ', 2))'

if __name__ == '__main__':
    # from args get model type, dataset type and testing configs
    args = parse_args()
    config_file = args.config

    # load config file
    config = load_yaml(config_file)

    res_list = [10, 20, 40, 80]

    for res in config["test_res_pair"]:
        model_dir = os.path.join(config["model_dir"], "res_{}_{}".format(res[0], res[1]), "model.pt")

        mesh_low = RectangleMesh(Point(0, 0), Point(1, 1), res_list[res[0]], res_list[res[0]])
        mesh_high = RectangleMesh(Point(0, 0), Point(1, 1), res_list[res[1]], res_list[res[1]])

        # solver parameters
        dt = 0.001
        T = 10
        initial_condition = Expression(gen_random_expression_str_2d(), degree=2)
        boundary_condition = [['Neumann', 0, 0]]

        # physical parameters
        nu = 0.01 

        solver_integrated = IntergratedBurgersSolver(model_dir, mesh_low, mesh_high, dt, T, nu, initial_condition, boundary_condition)
        solver_integrated.solve()

        solver_direct = BurgersSolver(mesh_high, mesh_high, dt, T, nu, initial_condition, boundary_condition)
        for i in range(0, int(T / dt)):
            solver_direct.solve(i = i)

