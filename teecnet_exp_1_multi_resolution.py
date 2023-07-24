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


def visualize_prediction(writer, data, model, epoch):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
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

    writer.add_figure("Prediction", fig, epoch)
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

    writer.add_figure("True", fig, epoch)
    plt.close(fig)

    temp_grid_error = np.abs(temp_grid - temp_grid_true)
    fig = plt.figure(figsize=(8, 6))
    plt.contourf(x_values, y_values, temp_grid_error, levels=np.linspace(0, 1, 100))
    # plt.contourf(x_values, y_values, temp_grid_error)
    plt.colorbar(label='Velocity Magnitude')
    plt.title('Velocity Error Map')
    plt.xlabel('x')
    plt.ylabel('y')

    writer.add_figure("Error", fig, epoch)
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

    writer.add_figure("Low Resolution", fig, epoch)
    plt.close(fig)



def train(model, dataset, log_dir, model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = initialize_model(type='NeuralOperator', in_channel=1, out_channel=1, width=64, ker_width=512, depth=6).to(device)
    model = model.to(device)
    print('The model has {} parameters'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    train_dataset, test_dataset = train_test_split(dataset, 0.8)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    writer = SummaryWriter(log_dir)

    os.makedirs(model_dir, exist_ok=True)
    t1 = time.time()
    for epoch in range(600):
        model.train()
        loss_all = 0
        accuracy_all = 0
        # i_sample = 0

        for data in train_loader:
            # model.train()
            # i_sample += 1
            # if i_sample > 200:
                # break

            data = data.to(device)
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            optimizer.zero_grad()
            out = model(x, edge_index, edge_attr)
            loss = torch.nn.functional.mse_loss(out, data.y)
            r2_accuracy = r2_score(data.y.cpu().detach().numpy(), out.cpu().detach().numpy())
            loss.backward()
            loss_all += loss.item()
            accuracy_all += r2_accuracy
            optimizer.step()

        scheduler.step()
        writer.add_scalar('Loss/train', loss_all / len(train_loader), epoch)
        writer.add_scalar('Accuracy/train', accuracy_all / len(train_loader), epoch)

        visualize_prediction(writer, data[0], model, epoch)

        print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss_all / len(train_loader)))

        if epoch % 10 == 0:
            model.eval()
            loss_all = 0
            for data in test_loader:
                data = data.to(device)
                x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
                out = model(x, edge_index, edge_attr)
                if data.y.dim() == 1:
                    data.y = data.y.unsqueeze(-1)

                loss = torch.nn.functional.mse_loss(out, data.y)
                loss_all += loss.item()
            writer.add_scalar('Loss/test', loss_all / len(test_loader), epoch)
            # torch.save(model.state_dict(), 'test_cases/burger/CFDError/{}/model_{}.pt'.format(sim_start_time, epoch))
            # print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss_all / len(test_loader)))
    t2 = time.time()
    print('Training time: {:.4f} s'.format(t2 - t1))
    torch.save(model.state_dict(), '{}/model.pt'.format(model_dir))
    writer.close()


def test(model, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = initialize_model(type='NeuralOperator', in_channel=1, out_channel=1, width=64, ker_width=512, depth=6).to(device)
    with torch.no_grad():
        model.eval()
        loss_all = 0
        accuracy_all = 0
        model.to(device)
        test_loader = DataLoader(dataset, batch_size=6, shuffle=False)

        for data in tqdm(test_loader):
            data = data.to(device)
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            out = model(x, edge_index, edge_attr)
            if data.y.dim() == 1:
                data.y = data.y.unsqueeze(-1)
            loss = torch.nn.functional.mse_loss(out, data.y)
            r2_accuracy = r2_score(data.y.cpu().detach().numpy(), out.cpu().detach().numpy())
            loss_all += loss.item()
            accuracy_all += r2_accuracy

        loss_all /= len(test_loader)
        accuracy_all /= len(test_loader)
        # print('resolution pair: {}_{}'.format(res_low, res_high))
        print('Loss: {:.4f}'.format(loss_all))
        print('Accuracy: {:.4f}'.format(accuracy_all))


if __name__ == '__main__':
    # from args get model type, dataset type and testing configs
    args = parse_args()
    config_file = args.config

    # load config
    config = load_yaml(config_file)
    
    # perform training on each individual train resolution pairs and save model
    for res in config["train_res_pair"]:
        # delete the processed dataset
        if os.path.exists(os.path.join(config["dataset_root"], "processed")):
            shutil.rmtree(os.path.join(config["dataset_root"], "processed"))
            
        dataset = initialize_dataset(dataset=config["dataset_type"], root=config["dataset_root"], res_low=res[0], res_high=res[1], pre_transform='interpolate_high')
        model = initialize_model(type=config["model_type"], in_channel=config["in_channel"], width=config["width"], out_channel=config["in_channel"], num_layers=config["num_layers"])

        log_dir = os.path.join(config["log_dir"], config["model_type"], config["dataset_type"], "res_{}_{}".format(res[0], res[1]))
        model_dir = os.path.join(config["model_dir"], config["model_type"], "res_{}_{}".format(res[0], res[1]))

        train(model, dataset, log_dir, model_dir)

    # perform validation on each individual test pairs
    for res_tr in config["train_test_pair"]:
        for res_te in config["test_res_pair"]:
            dataset = initialize_dataset(dataset=config["dataset_type"], root=config["dataset_root"], res_low=res[0], res_high=res[1], pre_transform='interpolate_high')
            model = initialize_model(type=config["model_type"], in_channel=1, width=16, out_channel=1, num_layers=3)

            model_dir = os.path.join(config["model_dir"], config["model_type"], config["dataset_type"], "res_{}_{}".format(res_tr[0], res_tr[1]))
            model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))
            print("Model trained on res pair: {}".format(res_tr) + "and tested on res pair: {}".format(res_te))
            test(model, dataset)
    
