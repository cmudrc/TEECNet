import os
import shutil
import time
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from sklearn.metrics import r2_score
# from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
from utils import train_test_split, get_cur_time, initialize_model, initialize_dataset, parse_args, load_yaml


def visualize_prediction(writer, data, model, epoch, mode='writer', **kwargs):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    x = x.to(kwargs['device'])
    edge_index = edge_index.to(kwargs['device'])
    edge_attr = edge_attr.to(kwargs['device'])

    pred = model(x, edge_index, edge_attr).detach().cpu().numpy()
    x = data.pos[:, 0].detach().cpu().numpy()
    y = data.pos[:, 1].detach().cpu().numpy()
    # x = data.pos[:, 0].detach().cpu().numpy()
    # y = data.pos[:, 1].detach().cpu().numpy()
    
    x_values = np.unique(x)
    y_values = np.unique(y)
    temp_grid = pred.squeeze().reshape(len(x_values), len(y_values))

    fig = plt.figure(figsize=(8, 6))
    plt.contourf(x_values, y_values, temp_grid, levels=np.linspace(0, 1, 100))
    # plt.contourf(x_values, y_values, temp_grid)
    plt.colorbar(label='Velocity Magnitude')
    plt.title('Velocity Contour Plot')
    plt.xlabel('x')
    plt.ylabel('y')

    if mode == 'writer':
        writer.add_figure("Prediction", fig, epoch)
    elif mode == 'save':
        save_dir = kwargs['save_dir']
        plt.savefig(os.path.join(save_dir, 'prediction.png'))
    plt.close(fig)

    temp_grid_true = data.y.cpu().detach().numpy().squeeze().reshape(len(x_values), len(y_values))
    fig = plt.figure(figsize=(8, 6))
    plt.contourf(x_values, y_values, temp_grid_true, levels=np.linspace(0, 1, 100))
    # plt.contourf(x_values, y_values, temp_grid_true)
    # limit the three figures to have the same colorbar
    plt.colorbar(label='Velocity Magnitude')
    plt.title('Velocity Contour Plot')
    plt.xlabel('x')
    plt.ylabel('y')

    if mode == 'writer':
        writer.add_figure("True", fig, epoch)
    elif mode == 'save':
        save_dir = kwargs['save_dir']
        plt.savefig(os.path.join(save_dir, 'true.png'))
    plt.close(fig)

    temp_grid_error = np.abs(temp_grid - temp_grid_true)
    fig = plt.figure(figsize=(8, 6))
    plt.contourf(x_values, y_values, temp_grid_error, levels=np.linspace(0, 1, 100))
    # plt.contourf(x_values, y_values, temp_grid_error)
    plt.colorbar(label='Velocity Magnitude')
    plt.title('Velocity Error Map')
    plt.xlabel('x')
    plt.ylabel('y')

    if mode == 'writer':
        writer.add_figure("Error", fig, epoch)
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

    fig = plt.figure(figsize=(8, 6))
    # plt.contourf(x_values_low, y_values_low, temp_grid_low, levels=np.linspace(0, 1, 100), cmap="RdBu_r")
    plt.contourf(x_values, y_values, temp_grid_low, levels=np.linspace(0, 1, 100))
    # plt.contourf(x_values, y_values, temp_grid_low)
    plt.colorbar(label='Velocity Magnitude')
    plt.title('Velocity Contour Map')   
    plt.xlabel('x')
    plt.ylabel('y')
    if mode == 'writer':
        writer.add_figure("Low Resolution", fig, epoch)
    plt.close(fig)

def validate_geometry(model, dataset, device, save_dir):
    model.eval()
    loader = DataLoader(dataset, batch_size=1)

    logger = SummaryWriter(save_dir)    

    loss_all = 0
    accuracy_all = 0

    for i in range(len(loader)):
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.edge_attr)
        if data.y.dim() == 1:
                data.y = data.y.unsqueeze(-1)
        loss = torch.nn.functional.mse_loss(pred, data.y)
        r2_accuracy = r2_score(data.y.cpu().detach().numpy(), pred.cpu().detach().numpy())
        loss_all += loss.item()
        accuracy_all += r2_accuracy

        logger.add_scalar('Loss', loss.item(), i)
        logger.add_scalar('R2 Accuracy', r2_accuracy, i)
        
        # save prediction every 50 steps
        save_dir_i = os.path.join(save_dir, '{}'.format(i))
        if not os.path.exists(save_dir_i):
            os.makedirs(save_dir_i)
        if i % 50 == 0:
            visualize_prediction(data, model, i, mode='save', device=device, save_dir=save_dir_i)

    return loss_all / len(loader), accuracy_all / len(loader)


if __name__ == '__main__':
    # from args get model type, dataset type and testing configs
    args = parse_args()
    config_file = args.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load config file
    config = load_yaml(config_file)

    for res in config["test_res_pair"]:
        # get model type and dataset type
        if os.path.exists(os.path.join(config["dataset_root"], "processed")):
            shutil.rmtree(os.path.join(config["dataset_root"], "processed"))
        dataset = initialize_dataset(dataset=config['dataset_type'], root=config['dataset_root'], res_low=res[0], res_high=res[1], pre_transform='interpolate_high')
        model = initialize_model(model=config['model_type'], in_channel=config['in_channel'], wdith=config['width'], out_channel=config['out_channel'], num_layers=config['num_layers'], retrieve_weight=False)
        model_dir = os.path.join(config["model_dir"], config["model_type"], "res_{}_{}".format(res[0], res[1]))
        model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))

        loss, accuracy = validate_geometry(model, dataset, device, model_dir)

        print("Resolution: {}x{}".format(res[0], res[1]))

        print("Loss: {}".format(loss))
        print("Accuracy: {}".format(accuracy))

        print("=====================================")

