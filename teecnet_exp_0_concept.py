####################################################################################################
# This script is used to perform experiment 0: concept proof of TEECNet
# The script will create simulations on low and high resolution dataset on different orders of shape functions, and then train a model on the created dataset.
# The coefficients of TEECNet trained on different orders of shape functions will be compared.


import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2' # Only add this for TRACE to work, comment out for other cases! 
import shutil
import time
import numpy as np
import torch
import wandb
from fenics import *
from dolfin import *
from fenicstools.Interpolation import interpolate_nonmatching_mesh

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from sklearn.metrics import r2_score
# from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import h5py
from utils import train_test_split, get_cur_time, initialize_model, initialize_dataset, parse_args, load_yaml


class HeatEquationDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, res_low=1, res_high=3, order_sf=1):
        self.res_list = [8, 16, 32, 64]
        self.order_sf = order_sf
        self.res_low = res_low
        self.res_high = res_high
        self.pre_transform = pre_transform
        # self.res_list = [10, 20, 40, 80]
        super(HeatEquationDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # enforce processing for all apllications
        self.process()

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)
    
    @property
    def is_processed(self):
        return False
    
    @property
    def raw_file_names(self):
        return None
    
    @property
    def mesh_file_names(self):
        return None

    @property
    def processed_file_names(self):
        return ['heat_transfer_data.pt']

    def process(self):
        data_list = []
        mesh_low = UnitSquareMesh(self.res_list[self.res_low], self.res_list[self.res_low])
        mesh_high = UnitSquareMesh(self.res_list[self.res_high], self.res_list[self.res_high])
        V = FunctionSpace(mesh_high, "CG", self.order_sf)
        coordinate = mesh_high.coordinates()
        edge_lists = edges(mesh_high)
        edge_index = np.zeros((2, 2 * mesh_high.num_edges()), dtype=np.int64)
        for i, edge in enumerate(edge_lists):
            edge_index[0, i] = edge.entities(0)[0]
            edge_index[1, i] = edge.entities(0)[1]
            edge_index[0, i + mesh_high.num_edges()] = edge.entities(0)[1]
            edge_index[1, i + mesh_high.num_edges()] = edge.entities(0)[0]
        edge_attr = np.concatenate([coordinate[edge_index[0]], coordinate[edge_index[1]], ], axis=1)
        
        for i in range(1000):
            random_heat_source = generate_random_heat_source()
            u_low = steady_state_heat_equation(self.order_sf, random_heat_source, mesh_low)
            u_high = steady_state_heat_equation(self.order_sf, random_heat_source, mesh_high)
            u_low = interpolate_nonmatching_mesh(u_low, V)
            u_low = u_low.compute_vertex_values(mesh_high)
            u_high = u_high.compute_vertex_values(mesh_high)

            u_low = u_low.reshape(-1, 1).astype(np.float32)
            u_high = u_high.reshape(-1, 1).astype(np.float32)

            data = Data(x=torch.from_numpy(u_low), y=torch.from_numpy(u_high), edge_index=torch.from_numpy(edge_index), edge_attr=torch.from_numpy(edge_attr).float(), pos=torch.from_numpy(coordinate).float())
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def visualize_prediction(data, model, epoch, mode='writer', **kwargs):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    x = x.to(kwargs['device'])
    edge_index = edge_index.to(kwargs['device'])
    edge_attr = edge_attr.to(kwargs['device'])

    pred = model(x, edge_index, edge_attr).detach().cpu().numpy()
    # pred = model(x, edge_index).detach().cpu().numpy() # for GCN
    x = data.pos[:, 0].detach().cpu().numpy()
    y = data.pos[:, 1].detach().cpu().numpy()
    # x = data.pos[:, 0].detach().cpu().numpy()
    # y = data.pos[:, 1].detach().cpu().numpy()
    
    x_values = np.unique(x)
    y_values = np.unique(y)
    temp_grid = pred.squeeze().reshape(len(x_values), len(y_values))

    fig = plt.figure(figsize=(12, 6))
    plt.contourf(x_values, y_values, temp_grid, levels=np.linspace(0, 1, 100))
    # plt.contourf(x_values, y_values, temp_grid)
    plt.colorbar(label='Velocity Magnitude')
    plt.title('Velocity Contour Plot')
    plt.xlabel('x')
    plt.ylabel('y')

    if mode == 'writer':
        wandb.log({"prediction": wandb.Image(plt)})
    elif mode == 'save':
        save_dir = kwargs['save_dir']
        plt.savefig(os.path.join(save_dir, 'prediction.png'))
    plt.close(fig)

    temp_grid_true = data.y.cpu().detach().numpy().squeeze().reshape(len(x_values), len(y_values))
    fig = plt.figure(figsize=(12, 6))
    plt.contourf(x_values, y_values, temp_grid_true, levels=np.linspace(0, 1, 100))
    # plt.contourf(x_values, y_values, temp_grid_true)
    # limit the three figures to have the same colorbar
    plt.colorbar(label='Velocity Magnitude')
    plt.title('Velocity Contour Plot')
    plt.xlabel('x')
    plt.ylabel('y')

    if mode == 'writer':
        wandb.log({"ground_truth": wandb.Image(plt)})
    elif mode == 'save':
        save_dir = kwargs['save_dir']
        plt.savefig(os.path.join(save_dir, 'true.png'))
    plt.close(fig)

    temp_grid_error = np.abs(temp_grid - temp_grid_true)
    fig = plt.figure(figsize=(12, 6))
    plt.contourf(x_values, y_values, temp_grid_error, levels=np.linspace(0, 1, 100))
    # plt.contourf(x_values, y_values, temp_grid_error)
    plt.colorbar(label='Velocity Magnitude')
    plt.title('Velocity Error Map')
    plt.xlabel('x')
    plt.ylabel('y')

    if mode == 'writer':
        wandb.log({"error": wandb.Image(plt)})
    elif mode == 'save':
        save_dir = kwargs['save_dir']
        plt.savefig(os.path.join(save_dir, 'error.png'))
    plt.close(fig)

    x_low = data.pos[:, 0].detach().cpu().numpy()
    y_low = data.pos[:, 1].detach().cpu().numpy()

    x_values_low = np.unique(x_low)
    y_values_low = np.unique(y_low)
    # temp_grid_low = data.x.detach().cpu().numpy().squeeze().reshape(len(x_values_low), len(y_values_low))
    temp_grid_low = data.x[:, 0].detach().cpu().numpy().squeeze().reshape(len(x_values), len(y_values))

    fig = plt.figure(figsize=(12, 6))
    # plt.contourf(x_values_low, y_values_low, temp_grid_low, levels=np.linspace(0, 1, 100), cmap="RdBu_r")
    plt.contourf(x_values, y_values, temp_grid_low, levels=np.linspace(0, 1, 100))
    # plt.contourf(x_values, y_values, temp_grid_low)
    plt.colorbar(label='Velocity Magnitude')
    plt.title('Velocity Contour Map')   
    plt.xlabel('x')
    plt.ylabel('y')
    if mode == 'writer':
        wandb.log({"low_resolution": wandb.Image(plt)})
    plt.close(fig)


def steady_state_heat_equation(order__sf, random_heat_source, mesh):
    # Define domain and mesh
    # xmin, xmax = 0, 1
    # ymin, ymax = 0, 1
    # mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), mesh_resolution, mesh_resolution)

    # Define function space
    V = FunctionSpace(mesh, "CG", order__sf)

    # Define boundary conditions
    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, Constant(0), boundary)

    x = SpatialCoordinate(mesh)
    heat_source = Expression(random_heat_source, sigma=0.1, degree=2)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    kappa = Constant(0.1)  # Thermal diffusivity
    F = kappa * inner(grad(u), grad(v)) * dx - heat_source * v * dx

    a, L = lhs(F), rhs(F)
    u = Function(V)

    # Solve the steady-state problem
    solve(a == L, u, bc)

    return project(u, V)


def generate_random_heat_source():
    source_x = np.random.uniform(0, 1)
    source_y = np.random.uniform(0, 1)

    return f"exp(-(pow((x[0] - {source_x}), 2) + pow((x[1] - {source_y}), 2))/(2*sigma*sigma))"


def train(model, dataset, model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = initialize_model(type='NeuralOperator', in_channel=1, out_channel=1, width=64, ker_width=512, depth=6).to(device)
    model = model.to(device)
    print('The model has {} parameters'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    train_dataset, test_dataset = train_test_split(dataset, 0.8)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    os.makedirs(model_dir, exist_ok=True)
    t1 = time.time()
    for epoch in range(200):
        model.train()
        loss_all = 0
        accuracy_all = 0
        # i_sample = 0

        for data in train_loader:
            # model.train()
            # i_sample += 1
            # if i_sample > 200:
                # break

            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            x = x.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            optimizer.zero_grad()
            out = model(x, edge_index, edge_attr)
            # out = model(x, edge_index) # for GCN

            # torch.onnx.export(model, (x, edge_index, edge_attr), '{}/model.onnx'.format(model_dir), input_names=['temperature', 'edge_index', 'discretization length'], output_names=['temperature'])
            loss = torch.nn.functional.mse_loss(out, data.y.to(device))
            r2_accuracy = r2_score(data.y.cpu().detach().numpy(), out.cpu().detach().numpy())
            loss.backward()
            loss_all += loss.item()
            accuracy_all += r2_accuracy
            optimizer.step()

            # delete x, edge_index, edge_attr, out, loss to save gpu memory
            del x, edge_index, edge_attr, out, loss

        scheduler.step()
        wandb.log({"loss": loss_all / len(train_loader), "accuracy": accuracy_all / len(train_loader)})

        if epoch % 10 == 0:
            visualize_prediction(data[0], model, epoch, mode='writer', device=device)

        # print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss_all / len(train_loader)))

        if epoch % 10 == 0:
            model.eval()
            loss_all = 0
            for data in test_loader:
                data = data.to(device)
                x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
                out = model(x, edge_index, edge_attr)
                # out = model(x, edge_index) # for GCN
                if data.y.dim() == 1:
                    data.y = data.y.unsqueeze(-1)

                loss = torch.nn.functional.mse_loss(out, data.y)
                loss_all += loss.item()
            wandb.log({"loss_test": loss_all / len(test_loader)})
            torch.save(model.state_dict(), '{}/model_{}.pt'.format(model_dir, epoch))
            cur_model = wandb.Artifact("model_{}".format(epoch), type="model")
            cur_model.add_file('{}/model_{}.pt'.format(model_dir, epoch))
            wandb.log_artifact(cur_model)
            # wandb.link_artifact(cur_model, "model_{}".format(epoch))
            # torch.save(model.state_dict(), 'test_cases/burger/CFDError/{}/model_{}.pt'.format(sim_start_time, epoch))
            # print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss_all / len(test_loader)))
    t2 = time.time()
    print('Training time: {:.4f} s'.format(t2 - t1))
    torch.save(model.state_dict(), '{}/model.pt'.format(model_dir))
    # save onnx model for visualization
    # torch.onnx.export(model, (x, edge_index, edge_attr), '{}/model.onnx'.format(model_dir), input_names=['temperature', 'edge_index', 'discretization length'], output_names=['temperature'])


if __name__ == '__main__':
    # from args get model type, dataset type and testing configs
    args = parse_args()
    config_file = args.config

    # load config
    config = load_yaml(config_file)

    # initialize wandb
    wandb.init(project="teecnet_exp_0_concept", config=config)

    # create a txt file to record test results
    os.makedirs(os.path.join(config["log_dir"], config["model_type"], config["dataset_type"]), exist_ok=True)
    sf_orders = [1, 2, 3, 4]
    
    for order in sf_orders:
        # delete processed data
        shutil.rmtree(os.path.join(config["dataset_root"], "processed"), ignore_errors=True)
        # log current order of shape functions
        wandb.log({"order_sf": order})

        # initialize dataset
        dataset = HeatEquationDataset(root=config["dataset_root"], res_low=config["res_low"], res_high=config["res_high"], order_sf=order)

        # initialize model
        model = initialize_model(type=config["model_type"], in_channel=config["in_channel"], width=config["width"], out_channel=config["out_channel"], num_layers=config["num_layers"], retrieve_weight=False, num_powers=config["num_powers"])

        # train model
        train(model, dataset, os.path.join(config["log_dir"], config["model_type"], config["dataset_type"], "res_{}_{}".format(config["res_low"], config["res_high"]), "order_{}".format(order)))

        # extract coefficients in model
        coefficient = model.kernel.kernel.conv_out.root_param.detach().cpu().numpy()

        # log mean coefficient
        wandb.log({"mean_coefficient": np.mean(coefficient, axis=1)})


