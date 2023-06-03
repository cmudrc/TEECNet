import os
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


NUM_FIXED_ALPHA_EPOCHS = 100
NUM_FIXED_COEFFICIENT_EPOCHS = 100


# function for debug purpose
def print_groups_and_datasets(name, obj):
    print(name, ":", type(obj))
    
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
    pred = model(data).detach().cpu().numpy()
    # x = data.pos_high[:, 0].detach().cpu().numpy()
    # y = data.pos_high[:, 1].detach().cpu().numpy()
    x = data.pos[:, 0].detach().cpu().numpy()
    y = data.pos[:, 1].detach().cpu().numpy()
    
    x_values = np.unique(x)
    y_values = np.unique(y)
    temp_grid = pred.squeeze().reshape(len(x_values), len(y_values))

    fig = plt.figure(figsize=(8, 6))
    plt.contourf(x_values, y_values, temp_grid, levels=15, cmap="RdBu_r")
    plt.colorbar(label='Temperature')
    plt.title('Temperature Contour Plot')
    plt.xlabel('x')
    plt.ylabel('y')

    writer.add_figure("Prediction", fig, epoch)
    plt.close(fig)

    temp_grid_true = data.y.cpu().detach().numpy().squeeze().reshape(len(x_values), len(y_values))
    fig = plt.figure(figsize=(8, 6))
    plt.contourf(x_values, y_values, temp_grid_true, levels=15, cmap="RdBu_r")
    plt.colorbar(label='Temperature')
    plt.title('Temperature Contour Plot')
    plt.xlabel('x')
    plt.ylabel('y')

    writer.add_figure("True", fig, epoch)
    plt.close(fig)

    x_low = data.pos[:, 0].detach().cpu().numpy()
    y_low = data.pos[:, 1].detach().cpu().numpy()

    x_values_low = np.unique(x_low)
    y_values_low = np.unique(y_low)
    temp_grid_low = data.x.detach().cpu().numpy().squeeze().reshape(len(x_values_low), len(y_values_low))

    fig = plt.figure(figsize=(8, 6))
    plt.contourf(x_values_low, y_values_low, temp_grid_low, levels=15, cmap="RdBu_r")
    plt.colorbar(label='Temperature')
    plt.title('Temperature Contour Plot')   
    plt.xlabel('x')
    plt.ylabel('y')

    writer.add_figure("Low Resolution", fig, epoch)
    plt.close(fig)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = HeatTransferNetwork(1, 64, 1, 2).to(device)
    model = initialize_model(type='HeatTransferNetwork', in_channel=1, hidden_channel=64, out_channel=1, num_kernels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    # dataset = HeatTransferDataset('dataset/heat', res_low=1, res_high=3)
    dataset = initialize_dataset(dataset='HeatTransferDataset', root='dataset/heat', res_low=0, res_high=3, pre_transform='interpolate')
    train_dataset, test_dataset = train_test_split(dataset, 0.8)
    train_loader = DataLoader(train_dataset, batch_size=36, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=36, shuffle=False)
    sim_start_time = get_cur_time()
    writer = SummaryWriter('runs/heat_transfer/CFDError/{}'.format(sim_start_time))

    os.makedirs('test_cases/heat_transfer/CFDError/{}'.format(sim_start_time), exist_ok=True)
    for epoch in range(1000):
        model.train()
        loss_all = 0
        # i_sample = 0

        for data in train_loader:
            # model.train()
            # i_sample += 1
            # if i_sample > 200:
                # break

            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
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

        visualize_alpha(writer, model, epoch)
        # visualize_coefficients(writer, model, epoch)
        visualize_clusters(writer, data, model, epoch)
        # visualize_errors_by_layer(writer, model, epoch)
        visualize_prediction(writer, data[0], model, epoch)

        print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss_all / len(train_loader)))

        if epoch % 10 == 0:
            model.eval()
            loss_all = 0
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                if data.y.dim() == 1:
                    data.y = data.y.unsqueeze(-1)
                loss = torch.nn.functional.mse_loss(out, data.y)
                loss_all += loss.item()
            writer.add_scalar('Loss/test', loss_all / len(test_loader), epoch)
            torch.save(model.state_dict(), 'test_cases/heat_transfer/CFDError/{}/model_{}.pt'.format(sim_start_time, epoch))
            print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss_all / len(test_loader)))

    torch.save(model.state_dict(), 'test_cases/heat_transfer/CFDError/{}/model.pt'.format(sim_start_time))
    writer.close()

def train_neural_op():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = initialize_model(type='NeuralOperator', in_channel=1, out_channel=1, width=11, ker_width=1024, depth=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    dataset = initialize_dataset('dataset/heat', res_low=0, res_high=3)
    train_dataset, test_dataset = train_test_split(dataset, 0.8)
    train_loader = DataLoader(train_dataset, batch_size=36, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=36, shuffle=False)
    sim_start_time = get_cur_time()
    writer = SummaryWriter('runs/heat_transfer/{}'.format(sim_start_time))

    os.makedirs('test_cases/heat_transfer/{}'.format(sim_start_time), exist_ok=True)
    for epoch in range(1000):
        model.train()
        loss_all = 0
        # i_sample = 0

        for data in train_loader:
            # model.train()
            # i_sample += 1
            # if i_sample > 200:
                # break

            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
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

        visualize_alpha(writer, model, epoch)
        # visualize_coefficients(writer, model, epoch)
        visualize_clusters(writer, data, model, epoch)
        # visualize_errors_by_layer(writer, model, epoch)
        visualize_prediction(writer, data[0], model, epoch)

        print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss_all / len(train_loader)))

        if epoch % 10 == 0:
            model.eval()
            loss_all = 0
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                if data.y.dim() == 1:
                    data.y = data.y.unsqueeze(-1)
                loss = torch.nn.functional.mse_loss(out, data.y)
                loss_all += loss.item()
            writer.add_scalar('Loss/test', loss_all / len(test_loader), epoch)
            torch.save(model.state_dict(), 'test_cases/heat_transfer/{}/model_{}.pt'.format(sim_start_time, epoch))
            print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss_all / len(test_loader)))

    torch.save(model.state_dict(), 'test_cases/heat_transfer/{}/model.pt'.format(sim_start_time))
    writer.close()


if __name__ == '__main__':
    train_neural_op()