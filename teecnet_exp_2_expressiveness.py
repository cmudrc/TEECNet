import os
import shutil
import time
import numpy as np
import torch
import wandb

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from sklearn.metrics import r2_score
# from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
from utils import train_test_split, initialize_model, initialize_dataset, parse_args, load_yaml


def plot_edge_attributes(edge_index, edge_attr, pos):
    num_edges = edge_index.shape[1]
    x_values = []
    y_values = []
    edge_values = []

    for i in range(num_edges):
        start_node = edge_index[0][i]
        end_node = edge_index[1][i]
        x_start, y_start = pos[start_node]
        x_end, y_end = pos[end_node]
        x_center = (x_start + x_end) / 2
        y_center = (y_start + y_end) / 2
        edge_value = edge_attr[i].item()
        x_values.append(x_center)
        y_values.append(y_center)
        edge_values.append(edge_value)

    x_values = np.array(x_values).squeeze()
    y_values = np.array(y_values).squeeze()
    edge_values = np.array(edge_values).squeeze()

    X, Y = np.meshgrid(np.unique(x_values), np.unique(y_values))
    edge_value = np.zeros((len(np.unique(x_values)), len(np.unique(y_values))))
    for i in range(len(x_values)):
        x_index = np.where(X == x_values[i])
        y_index = np.where(Y == y_values[i])
        edge_value[x_index, y_index] = edge_values[i]

    fig = plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, edge_value)
    plt.title('Edge Attributes')
    plt.colorbar()
    return fig


def visualize_prediction(data, model, epoch, **kwargs):
    model.eval()
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    x = x.to(kwargs['device'])
    edge_index = edge_index.to(kwargs['device'])
    edge_attr = edge_attr.to(kwargs['device'])
    pred = model(x, edge_index, edge_attr).detach().cpu().numpy()
    # pred = model(x, edge_index).detach().cpu().numpy()
    x = data.pos[:, 0].detach().cpu().numpy()
    y = data.pos[:, 1].detach().cpu().numpy()
    # x = data.pos[:, 0].detach().cpu().numpy()
    # y = data.pos[:, 1].detach().cpu().numpy()
    
    x_values = np.unique(x)
    y_values = np.unique(y)
    temp_grid = pred.squeeze().reshape(len(x_values), len(y_values))
    # temp_grid = np.sqrt(temp_grid[:, 0] ** 2 + temp_grid[:, 1] ** 2).reshape(len(x_values), len(y_values))

    fig = plt.figure(figsize=(8, 6))
    plt.contourf(x_values, y_values, temp_grid, levels=100, cmap='jet')
    # plt.contourf(x_values, y_values, temp_grid)
    plt.colorbar()
    # plt.title('Velocity Contour Plot')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.axes('off')
    # remove the axis
    plt.axis('off')
    wandb.log({"Prediction": wandb.Image(fig)})
    # plt.savefig("figures/pred_{}.png".format(epoch))
    # writer.add_figure("Prediction", fig, epoch)
    plt.close(fig)

    temp_grid_true = data.y.cpu().detach().numpy().squeeze().reshape(len(x_values), len(y_values))
    # temp_grid_true = np.sqrt(temp_grid_true[:, 0] ** 2 + temp_grid_true[:, 1] ** 2).reshape(len(x_values), len(y_values))
    fig = plt.figure(figsize=(8, 6))
    plt.contourf(x_values, y_values, temp_grid_true, levels=np.linspace(0, 1, 100), cmap='jet')
    # plt.contourf(x_values, y_values, temp_grid_true)
    # limit the three figures to have the same colorbar
    plt.colorbar()
    # plt.axes('off')
    # remove the axis   
    plt.axis('off')
    wandb.log({"True": wandb.Image(fig)})
    # plt.title('Velocity Contour Plot')
    # plt.xlabel('x')
    # plt.ylabel('y')

    # writer.add_figure("True", fig, epoch)
    plt.close(fig)

    temp_grid_error = np.abs(temp_grid - temp_grid_true)
    fig = plt.figure(figsize=(8, 6))
    plt.contourf(x_values, y_values, temp_grid_error, levels=np.linspace(0, 1, 100), cmap='jet')
    # plt.contourf(x_values, y_values, temp_grid_error)
    plt.colorbar()
    # plt.axes('off')
    # remove the axis
    plt.axis('off')
    wandb.log({"Error": wandb.Image(fig)})
    # plt.title('Velocity Error Map')
    # plt.xlabel('x')
    # plt.ylabel('y')

    # writer.add_figure("Error", fig, epoch)
    plt.close(fig)

    x_low = data.pos[:, 0].detach().cpu().numpy()
    y_low = data.pos[:, 1].detach().cpu().numpy()

    x_values_low = np.unique(x_low)
    y_values_low = np.unique(y_low)
    # temp_grid_low = data.x.detach().cpu().numpy().squeeze().reshape(len(x_values_low), len(y_values_low))
    temp_grid_low = data.x[:, 0].detach().cpu().numpy().squeeze().reshape(len(x_values_low), len(y_values_low))
    # temp_grid_low = np.sqrt(temp_grid_low[:, 0] ** 2 + temp_grid_low[:, 1] ** 2).reshape(len(x_values_low), len(y_values_low))

    fig = plt.figure(figsize=(8, 6))
    # plt.contourf(x_values_low, y_values_low, temp_grid_low, levels=np.linspace(0, 1, 100), cmap="RdBu_r")
    plt.contourf(x_values, y_values, temp_grid_low, levels=np.linspace(0, 1, 100), cmap="jet")
    # plt.contourf(x_values, y_values, temp_grid_low)
    plt.colorbar()
    # plt.axes('off')
    # remove the axis
    plt.axis('off')
    # plt.title('Velocity Contour Map')   
    # plt.xlabel('x')
    # plt.ylabel('y')

    # writer.add_figure("Low Resolution", fig, epoch)
    plt.close(fig)

    # kernel_k = model.kernel_out.weight_k.detach().cpu().numpy().squeeze()
    # kernel_op = model.kernel_out.weight_op.detach().cpu().numpy().squeeze()
    
    # fig_k = plot_edge_attributes(edge_index, kernel_k, data.pos)
    # writer.add_figure("Kernel_k", fig_k, epoch)
    # plt.close(fig_k)

    # fig_op = plot_edge_attributes(edge_index, kernel_op, data.pos)
    # writer.add_figure("Kernel_op", fig_op, epoch)
    # plt.close(fig_op)

    model.train()


def train(model, dataset, log_dir, model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = initialize_model(type='NeuralOperator', in_channel=1, out_channel=1, width=64, ker_width=512, depth=6).to(device)
    model = model.to(device)
    print('The model has {} parameters'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    # train_dataset, test_dataset = train_test_split(dataset, 0.8)
    train_dataset = dataset[:int(len(dataset) * 0.9)]
    test_dataset = dataset[int(len(dataset) * 0.9):]
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)
    # select one sample from the test dataset to visualize
    test_data = test_dataset[10]

    os.makedirs(model_dir, exist_ok=True)
    t1 = time.time()
    for epoch in range(1):
        model.train()
        loss_all = 0
        accuracy_all = 0
        i_sample = 0

        for data in train_loader:
            model.train()
            # i_sample += 1
            # if i_sample > 160:
            #     break

            data = data.to(device)
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            optimizer.zero_grad()
            out = model(x, edge_index, edge_attr)
            # out = model(x, edge_index)
            loss = torch.nn.functional.mse_loss(out, data.y)
            loss_by_sample = torch.nn.functional.mse_loss(out, data.y, reduction='none')
            r2_accuracy = r2_score(data.y.cpu().detach().numpy(), out.cpu().detach().numpy())
            loss.backward()
            wandb.log({"loss": loss, "r2_accuracy": r2_accuracy})
            wandb.log({"loss_by_sample": wandb.Histogram(loss_by_sample.cpu().detach().numpy())})

            # writer.add_scalar('Loss/train', loss, i_sample)
            # writer.add_scalar('Accuracy/train', r2_accuracy, i_sample)
            loss_all += loss.item()
            accuracy_all += r2_accuracy
            optimizer.step()

        # wandb.log({"loss_train": loss_all / len(train_loader), "r2_accuracy_train": accuracy_all / len(train_loader)})
            visualize_prediction(test_data, model, epoch, device=device)

        model.eval()
        with torch.no_grad():
            loss_all_test = []
            for data in test_loader:
                data = data.to(device)
                x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
                out = model(x, edge_index, edge_attr)
                # out = model(x, edge_index)
                if data.y.dim() == 1:
                    data.y = data.y.unsqueeze(-1)

                loss = torch.nn.functional.mse_loss(out, data.y)
                loss_all_test.append(loss.item())

        wandb.log({"loss_test": loss_all_test})

        scheduler.step()

        print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss_all / len(train_loader)))
        # writer.add_scalar('Loss/test', loss_all / len(test_loader), epoch)
        # torch.save(model.state_dict(), 'test_cases/burger/CFDError/{}/model_{}.pt'.format(sim_start_time, epoch))
        # print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss_all / len(test_loader)))
    t2 = time.time()
    print('Training time: {:.4f} s'.format(t2 - t1))
    torch.save(model.state_dict(), '{}/model.pt'.format(model_dir))


if __name__ == '__main__':
    # from args get model type, dataset type and testing configs
    args = parse_args()
    config_file = args.config

    # load config
    config = load_yaml(config_file)

    # initialize wandb
    wandb.init(project="teecnet_exp_2_expressiveness", config=config)

    res = config["train_res"]
    
    # # delete the processed dataset
    # if os.path.exists(os.path.join(config["dataset_root"], "processed")):
    #     shutil.rmtree(os.path.join(config["dataset_root"], "processed"))
        
    dataset = initialize_dataset(dataset=config["dataset_type"], root=config["dataset_root"], res_low=res[0], res_high=res[1], pre_transform='interpolate_high')
    model = initialize_model(type=config["model_type"], in_channel=config["in_channel"], width=config["width"], out_channel=config["in_channel"], num_layers=config["num_layers"], retrieve_weight=True, num_powers=config["num_powers"])

    log_dir = os.path.join(config["log_dir"], config["model_type"], config["dataset_type"], "res_{}_{}".format(res[0], res[1]))
    model_dir = os.path.join(config["model_dir"], config["model_type"], "res_{}_{}".format(res[0], res[1]))

    train(model, dataset, log_dir, model_dir)
