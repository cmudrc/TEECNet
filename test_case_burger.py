import os
import time
import numpy as np
import torch
from scipy.interpolate import griddata

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
# import torch.nn.functional as F
# from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
from utils import train_test_split, get_cur_time, initialize_model, initialize_dataset


def visualize_alpha(writer, model, epoch):
    alphas = model.alpha
    # alphas = np.array(alphas, dtype=np.float32)
    num_order = len(alphas[1][0])
    for i in range(num_order):
        writer.add_histogram(f"Alpha Order {i}", alphas[1][:, i], epoch)

def visualize_coefficients(writer, model, epoch):
    coefficients = model.coefficient[1]
    # coefficients = coefficients.detach().cpu().numpy()
    writer.add_histogram("Coefficients", coefficients, epoch)

def visualize_errors_by_layer(writer, model, epoch):
    errors = model.errors
    for i, error in enumerate(errors):
        # error = error.detach().cpu().numpy()
        writer.add_histogram(f"Error Layer {i}", error, epoch)

def visualize_clusters(writer, data, model, epoch):
    clusters = model.cluster
    # clusters = clusters.detach().cpu().numpy()
    fig = plt.figure()
    plt.scatter(data.pos[:, 0].detach().cpu().numpy(), data.pos[:, 1].detach().cpu().numpy(), c=clusters.detach().cpu().numpy(), cmap="viridis")
    plt.colorbar()
    plt.title(f"Clusters (Epoch: {epoch})")

    writer.add_figure("Clusters", fig, epoch)
    plt.close(fig)

def visualize_prediction(writer, data, model, epoch):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    pred = model(x, edge_index, edge_attr).detach().cpu().numpy()
    x = data.pos_high[:, 0].detach().cpu().numpy()
    y = data.pos_high[:, 1].detach().cpu().numpy()
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



def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = initialize_model(type='TEECNet', in_channel=1, width=16, out_channel=1, num_layers=3).to(device)
    # model = initialize_model(type='NeuralOperator', in_channel=1, out_channel=1, width=64, ker_width=512, depth=6).to(device)
    print('The model has {} parameters'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    dataset = initialize_dataset(dataset='BurgersDataset', root='dataset/burger', res_low=0, res_high=1, pre_transform='interpolate_high')
    train_dataset, test_dataset = train_test_split(dataset, 0.8)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    sim_start_time = get_cur_time()
    writer = SummaryWriter('runs/burger/CFDError/{}'.format(sim_start_time))

    os.makedirs('test_cases/burger/CFDError/{}'.format(sim_start_time), exist_ok=True)
    t1 = time.time()
    for epoch in range(5000):
        model.train()
        loss_all = 0
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
            # if data.y.dim() == 1:
            #         data.y = data.y.unsqueeze(-1)

            loss = torch.nn.functional.mse_loss(out, data.y)
            # r2_accuracy = r2_score(data.y.cpu().detach().numpy(), out.cpu().detach().numpy())
            loss.backward()
            loss_all += loss.item()
            optimizer.step()

            # following code evaluates the model performance with each training sample
            # if i_sample in [1, 2, 5, 50, 200]:
            #     model.eval()
            #     with torch.no_grad():
            #         data = test_dataset[np.random.randint(len(test_dataset))]
            #         data = data.to(device)
            #         out = model(data)
            #         if data.y.dim() == 1:
            #             data.y = data.y.unsqueeze(-1)

            #         loss = torch.nn.functional.mse_loss(out, data.y)
                    # r2_accuracy = r2_score(data.y.cpu().detach().numpy(), out.cpu().detach().numpy())
                    
                    # writer.add_scalar('Loss/test', loss, i_sample)
                    # writer.add_scalar('R2 Accuracy/test', r2_accuracy, i_sample)
                    # visualize_prediction(writer, data, model, i_sample)
                    # visualize_alpha(writer, model, i_sample)


        scheduler.step()
        writer.add_scalar('Loss/train', loss_all / len(train_loader), epoch)

        # visualize_alpha(writer, model, epoch)
        # visualize_coefficients(writer, model, epoch)
        # visualize_clusters(writer, data, model, epoch)
        # visualize_errors_by_layer(writer, model, epoch)
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
            torch.save(model.state_dict(), 'test_cases/burger/CFDError/{}/model_{}.pt'.format(sim_start_time, epoch))
            print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss_all / len(test_loader)))
    t2 = time.time()
    print('Training time: {:.4f} s'.format(t2 - t1))
    torch.save(model.state_dict(), 'test_cases/burger/CFDError/{}/model.pt'.format(sim_start_time))
    writer.close()


if __name__ == '__main__':
    train()