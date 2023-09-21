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
    # pred = model(x, edge_index).detach().cpu().numpy() # for GCN
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


def train(model, dataset, log_dir, model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = initialize_model(type='NeuralOperator', in_channel=1, out_channel=1, width=64, ker_width=512, depth=6).to(device)
    model = model.to(device)
    print('The model has {} parameters'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    train_dataset, test_dataset = train_test_split(dataset, 0.8)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    writer = SummaryWriter(log_dir)

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

        scheduler.step()
        writer.add_scalar('Loss/train', loss_all / len(train_loader), epoch)
        writer.add_scalar('Accuracy/train', accuracy_all / len(train_loader), epoch)

        if epoch % 50 == 0:
            visualize_prediction(writer, data[0], model, epoch, mode='writer', device=device)

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
            writer.add_scalar('Loss/test', loss_all / len(test_loader), epoch)
            # torch.save(model.state_dict(), 'test_cases/burger/CFDError/{}/model_{}.pt'.format(sim_start_time, epoch))
            # print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss_all / len(test_loader)))
    t2 = time.time()
    print('Training time: {:.4f} s'.format(t2 - t1))
    torch.save(model.state_dict(), '{}/model.pt'.format(model_dir))
    # save onnx model for visualization
    # torch.onnx.export(model, (x, edge_index, edge_attr), '{}/model.onnx'.format(model_dir), input_names=['temperature', 'edge_index', 'discretization length'], output_names=['temperature'])
    writer.close()


def test(model, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # model = initialize_model(type='NeuralOperator', in_channel=1, out_channel=1, width=64, ker_width=512, depth=6).to(device)
    with torch.no_grad():
        model.eval()
        loss_all = []
        accuracy_all = []
        model.to(device)
        test_loader = DataLoader(dataset, batch_size=6, shuffle=False)

        for data in test_loader:
            data = data.to(device)
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            x = x.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            out = model(x, edge_index, edge_attr)
            # out = model(x, edge_index) # for GCN
            # torch.onnx.export(model, (x, edge_index, edge_attr), '{}/model.onnx'.format(model_dir), input_names=['temperature', 'edge_index', 'discretization length'], output_names=['temperature'])
            if data.y.dim() == 1:
                data.y = data.y.unsqueeze(-1)
            loss = torch.nn.functional.mse_loss(out, data.y)
            r2_accuracy = r2_score(data.y.cpu().detach().numpy(), out.cpu().detach().numpy())
            loss_all.append(loss.item())
            accuracy_all.append(r2_accuracy)
        
        # visualize one sample
        image_save_dir = os.path.join(config["log_dir"], config["model_type"], config["dataset_type"], "res_{}_{}".format(res_tr[0], res_tr[1]), "res_{}_{}".format(res_te[0], res_te[1]))
        os.makedirs(image_save_dir, exist_ok=True)
        visualize_prediction(None, data[0], model, 0, mode='save', save_dir=image_save_dir, device=device)

        loss_all = np.array(loss_all).sum() / len(test_loader)
        loss_all_std = np.array(loss_all).std()
        accuracy_all = np.array(accuracy_all).sum() / len(test_loader)
        accuracy_all_std = np.array(accuracy_all).std()
        # print('resolution pair: {}_{}'.format(res_low, res_high))
        return loss_all, accuracy_all, loss_all_std, accuracy_all_std


if __name__ == '__main__':
    # from args get model type, dataset type and testing configs
    args = parse_args()
    config_file = args.config

    # load config
    config = load_yaml(config_file)

    # create a txt file to record test results
    os.makedirs(os.path.join(config["log_dir"], config["model_type"], config["dataset_type"]), exist_ok=True)
    with open(os.path.join(config["log_dir"], config["model_type"], config["dataset_type"], "test_results.txt"), "w") as f:    
        # perform training on each individual train resolution pairs and save model
        for res in config["train_res_pair"]:
            # delete the processed dataset
            if os.path.exists(os.path.join(config["dataset_root"], "processed")):
                shutil.rmtree(os.path.join(config["dataset_root"], "processed"))
                
            dataset = initialize_dataset(dataset=config["dataset_type"], root=config["dataset_root"], res_low=res[0], res_high=res[1], pre_transform='interpolate_high')
            model = initialize_model(type=config["model_type"], in_channel=config["in_channel"], width=config["width"], out_channel=config["out_channel"], num_layers=config["num_layers"], retrieve_weight=False)

            log_dir = os.path.join(config["log_dir"], config["model_type"], config["dataset_type"], "res_{}_{}".format(res[0], res[1]))
            model_dir = os.path.join(config["model_dir"], config["model_type"], "res_{}_{}".format(res[0], res[1]))

            train(model, dataset, log_dir, model_dir)

        # perform validation on each individual test pairs
        for res_tr in config["train_res_pair"]:
            for res_te in config["test_res_pair"]:
                # delete the processed dataset
                if os.path.exists(os.path.join(config["dataset_root"], "processed")):
                    shutil.rmtree(os.path.join(config["dataset_root"], "processed"))
                dataset = initialize_dataset(dataset=config["dataset_type"], root=config["dataset_root"], res_low=res_te[0], res_high=res_te[1], pre_transform='interpolate_high')
                model = initialize_model(type=config["model_type"], in_channel=config["in_channel"], width=config["width"], out_channel=config["out_channel"], num_layers=config["num_layers"], retrieve_weight=False)

                model_dir = os.path.join(config["model_dir"], config["model_type"], "res_{}_{}".format(res_tr[0], res_tr[1]))
                model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))
                # print("Model trained on res pair: {}".format(res_tr) + "and tested on res pair: {}".format(res_te))
                f.write("Model trained on res pair: {}".format(res_tr) + "and tested on res pair: {}".format(res_te) + "\n")
                loss, accuracy, loss_std, accuracy_std = test(model, dataset)
                # print("Loss: {:.4f}".format(loss))
                # print("Accuracy: {:.4f}".format(accuracy))
                f.write("Loss: {:.4f}+-{:.4f}".format(loss, loss_std) + "\n")
                f.write("Accuracy: {:.4f}+-{:.4f}".format(accuracy, accuracy_std) + "\n")
    
