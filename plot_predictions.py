import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import torch
from utils import *
import meshio


def save_prediction_results(data, mesh, model, model_dir, save_path=None):
    """
    Takes las input data and trained model to plot prediction results
    :param las_dir: directory of las data
    :param has_dir: directory of has data
    :param model: initialized model
    :param model_dir: directory of trained model
    :param save_path: directory to save the plot
    """
    device = torch.device('cpu')
    checkpoint_load(model, model_dir)
    model = model.to(device)
    model.eval()

    (data_l, data_h) = data
    data_l = data_l.to(device)
    data_h = data_h.to(device)
    has = model(data_l, data_h)
    prediction = has.detach().numpy()
    low_res = data_l.x.detach().numpy()
    high_res = data_h.x.detach().numpy()

    u_has_star = [prediction[:, 0], prediction[:, 1]]
    p_has_star = prediction[:, 2]

    u_has = [high_res[:, 0], high_res[:, 1]]
    p_has = high_res[:, 2]

    u_las = [low_res[:, 0], low_res[:, 1]]
    p_las = low_res[:, 2]

    np.savez(save_path, u_has_star=u_has_star, p_has_star=p_has_star, u_has=u_has, p_has=p_has, u_las=u_las, p_las=p_las)

    # plot contour map of u_has_star and u_has based on mesh
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set_title("u_has_star")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # reconstruct cells using edges
    u_has_star_absolute = np.sqrt(u_has_star[0] ** 2 + u_has_star[1] ** 2)
    points = mesh.points
    cells = mesh.get_cells_type("triangle")
    triang = tri.Triangulation(points[:, 0], points[:, 1], cells)

    contour = ax.tricontourf(triang, u_has_star_absolute, 100, cmap="jet")
    fig.colorbar(contour, ax=ax, label="absolute velocity - prediction")

    # plot contour map of u_has based on mesh alongside
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set_title("u_has")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # reconstruct cells using edges
    u_has_absolute = np.sqrt(u_has[0] ** 2 + u_has[1] ** 2)
    points = mesh.points    
 
    contour = ax.tricontourf(triang, u_has_absolute, 100, cmap="jet")
    fig.colorbar(contour, ax=ax, label="absolute velocity - ground truth")

    # plot quiver map of u_has_star and u_has based on mesh
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set_title("u_has_star")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.quiver(points[:, 0], points[:, 1], u_has_star[0], u_has_star[1], cmap='jet', scale=100)

    # plot quiver map of u_has based on mesh alongside
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set_title("u_has")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.quiver(points[:, 0], points[:, 1], u_has[0], u_has[1], cmap='jet', scale=100)
    plt.show()
    


if __name__ == "__main__":
    train_config = load_yaml("config/train_config.yaml")
    model = initialize_model(in_channel=train_config["in_channel"], out_channel=train_config["out_channel"], type=train_config["model_name"], layers=None, num_filters=None)
    dataset = initialize_dataset(dataset="MegaFlow2D", split_scheme=train_config["split_scheme"], dir=train_config["data_dir"], transform=train_config["transform"], split_ratio=train_config["split_ratio"], pre_transform=None)
    data_name, data = dataset.get_eval(203688)
    str1, str2, str3 = data_name.split("_")
    mesh_name = str1 + "_" + str2
    print(data_name)
    mesh = meshio.read("D:/Work/research/train/plot_mesh_structure/{}.msh".format(mesh_name))

    save_prediction_results(data, mesh, model, model_dir="D:/Work/research/train/checkpoints/2023-03-25_11-38/checkpoint-000050.pth", save_path="D:/Work/research/train/prediction/prediction_result_{}_model_050.npz".format(data_name))