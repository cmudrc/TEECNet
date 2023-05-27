import os
import numpy as np
import torch
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
    clusters = model.cluster[1]
    # clusters = clusters.detach().cpu().numpy()
    fig = plt.figure()
    plt.scatter(data.pos[:, 0].detach().cpu().numpy(), data.pos[:, 1].detach().cpu().numpy(), c=clusters.detach().cpu().numpy(), cmap="viridis")
    plt.colorbar()
    plt.title(f"Clusters (Epoch: {epoch})")

    writer.add_figure("Clusters", fig, epoch)
    plt.close(fig)

def visualize_prediction(writer, data, model, epoch):
    pred = model(data)
    X = data.pos[:, 0].detach().cpu().numpy()
    Y = data.pos[:, 1].detach().cpu().numpy()
    # reconstruct triangular element via edge_index
    tri_idx = data.edge_index.cpu().numpy().T
    tri = tri.Triangulation(X, Y, triangles=tri_idx)
    # plot prediction
    fig = plt.figure()
    plt.tricontourf(tri, pred.detach().cpu().numpy().flatten())
    plt.colorbar()
    plt.title(f'Prediction (Epoch: {epoch})')
    writer.add_figure("Prediction", fig, epoch)
    plt.close(fig)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = HeatTransferNetwork(1, 64, 1, 2).to(device)
    model = initialize_model(type='HeatTransferNetwork', in_channel=1, hidden_channel=64, out_channel=1, num_kernels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    # dataset = HeatTransferDataset('dataset/heat', res_low=1, res_high=3)
    dataset = initialize_dataset(dataset='HeatTransferDataset', root='dataset/heat', res_low=1, res_high=3)
    train_dataset, test_dataset = train_test_split(dataset, 0.8)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    sim_start_time = get_cur_time()
    writer = SummaryWriter('runs/heat_transfer/{}'.format(sim_start_time))

    os.makedirs('test_cases/heat_transfer/{}'.format(sim_start_time), exist_ok=True)
    for epoch in range(1):
        # model.train()
        loss_all = 0
        i_sample = 0

        for data in train_loader:
            model.train()
            i_sample += 1
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            if data.y.dim() == 1:
                    data.y = data.y.unsqueeze(-1)

            loss = torch.nn.functional.mse_loss(out, data.y)
            # r2_accuracy = r2_score(data.y.cpu().detach().numpy(), out.cpu().detach().numpy())
            loss.backward()
            loss_all += loss.item()
            optimizer.step()

            # following code evaluates the model performance with each training sample
            model.eval()
            with torch.no_grad():
                data = test_dataset[np.random.randint(len(test_dataset))]
                out = model(data)
                if data.y.dim() == 1:
                    data.y = data.y.unsqueeze(-1)

                loss = torch.nn.functional.mse_loss(out, data.y)
                # r2_accuracy = r2_score(data.y.cpu().detach().numpy(), out.cpu().detach().numpy())
                
                writer.add_scalar('Loss/test', loss, i_sample)
                # writer.add_scalar('R2 Accuracy/test', r2_accuracy, i_sample)
                visualize_prediction(writer, data, model, i_sample)


        scheduler.step()
        writer.add_scalar('Loss/train', loss_all / len(train_loader), epoch)

        try:
            visualize_alpha(writer, model, epoch)
            # visualize_coefficients(writer, model, epoch)
            visualize_clusters(writer, data, model, epoch)
            visualize_errors_by_layer(writer, model, epoch)
        except:
            pass

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
    train()