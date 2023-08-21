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
from matplotlib.tri import Triangulation
import h5py
from utils import train_test_split, get_cur_time, initialize_model, initialize_dataset, parse_args, load_yaml


def visualize_prediction(writer, data, model, epoch, mode='writer', **kwargs):
    x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y
    x = x.to(kwargs['device'])
    edge_index = edge_index.to(kwargs['device'])
    edge_attr = edge_attr.to(kwargs['device'])

    pred = model(x, edge_index, edge_attr).detach().cpu().numpy().squeeze()
    pos_x = data.pos[:, 0].detach().cpu().numpy()
    pos_y = data.pos[:, 1].detach().cpu().numpy()

    x = x.detach().cpu().numpy().squeeze()
    y = y.detach().cpu().numpy().squeeze()

    # reconstruct the mesh
    tri = Triangulation(pos_x, pos_y, data.cells.detach().cpu().numpy())
    # for debug purpose print triangulation x and y array shape
    # print(tri.x.shape)
    # print(tri.y.shape)
    # print(pred.shape)
    # print(tri.triangles.shape)
    # plot the temepreture contour
    plt.tricontourf(tri, pred, levels=np.linspace(0, 1, 100))
    plt.colorbar()
    plt.title('Prediction')
    
    if mode == 'writer':
        writer.add_figure('Prediction', plt.gcf(), epoch)
    elif mode == 'save':
        plt.savefig(os.path.join(kwargs['save_dir'], 'prediction_{}.png'.format(epoch)))
    
    plt.close()

    plt.tricontourf(tri, y, levels=np.linspace(0, 1, 100))
    plt.colorbar()
    plt.title('Ground Truth')
    if mode == 'writer':
        writer.add_figure('Ground Truth', plt.gcf(), epoch)
    elif mode == 'save':
        plt.savefig(os.path.join(kwargs['save_dir'], 'ground_truth_{}.png'.format(epoch)))
    plt.close()

    plt.tricontourf(tri, np.abs(pred - y), levels=np.linspace(0, 1, 100))
    plt.colorbar()
    plt.title('Absolute Error')
    if mode == 'writer':
        writer.add_figure('Absolute Error', plt.gcf(), epoch)
    elif mode == 'save':
        plt.savefig(os.path.join(kwargs['save_dir'], 'absolute_error_{}.png'.format(epoch)))

    plt.close()

    plt.tricontourf(tri, x, levels=np.linspace(0, 1, 100))
    plt.colorbar()
    plt.title('Low Resolution Temperature')
    if mode == 'writer':
        writer.add_figure('Low Resolution Temperature', plt.gcf(), epoch)
    elif mode == 'save':
        plt.savefig(os.path.join(kwargs['save_dir'], 'low_res_temperature_{}.png'.format(epoch)))

    plt.close()

    # free cuda memory
    del x, edge_index, edge_attr, y, pred, pos_x, pos_y, tri
    

def validate_geometry(model, dataset, device, save_dir):
    model.eval()
    loader = DataLoader(dataset, batch_size=1)

    logger = SummaryWriter(save_dir)    

    loss_all = 0
    accuracy_all = 0
    i = 0

    for data in tqdm(loader):
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
            visualize_prediction(logger, data, model, i, mode='save', device=device, save_dir=save_dir_i)
        i += 1
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
        model = initialize_model(type=config['model_type'], in_channel=config['in_channel'], width=config['width'], out_channel=config['out_channel'], num_layers=config['num_layers'], retrieve_weight=False)
        # model_dir = os.path.join(config["model_dir"], config["model_type"], "res_{}_{}".format(res[0], res[1]))
        save_dir = os.path.join(config["log_dir"], "res_{}_{}".format(res[0], res[1]))
        model.load_state_dict(torch.load(os.path.join(config['model_dir'], "model.pt")))
        model.to(device)
        loss, accuracy = validate_geometry(model, dataset, device, save_dir)

        print("Resolution: {}x{}".format(res[0], res[1]))

        print("Loss: {}".format(loss))
        print("Accuracy: {}".format(accuracy))

        print("=====================================")

