import os
import time
from utils import train_test_split, get_cur_time, initialize_model, initialize_dataset, parse_args, load_yaml
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.collections import LineCollection
import numpy as np
import shutil
import torch

# # plot heat solutions for all four resolutions
# if os.path.exists(os.path.join("dataset/heat", "processed")):
#     shutil.rmtree(os.path.join("dataset/heat", "processed"))

# dataset = initialize_dataset(dataset="HeatTransferDataset", root="dataset/heat", res_low=0, res_high=1, pre_transform='interpolate_high')
# data = dataset[0]
# pos_x = data.pos[:, 0]
# pos_y = data.pos[:, 1]

# x = data.x.numpy().squeeze()
# y = data.y.numpy().squeeze()

# x_values = np.unique(pos_x)
# y_values = np.unique(pos_y)

# temp_grid1 = x.reshape(len(x_values), len(y_values))
# temp_grid2 = y.reshape(len(x_values), len(y_values))

# if os.path.exists(os.path.join("dataset/heat", "processed")):
#     shutil.rmtree(os.path.join("dataset/heat", "processed"))

# dataset = initialize_dataset(dataset="HeatTransferDataset", root="dataset/heat", res_low=2, res_high=3, pre_transform='interpolate_high')
# data = dataset[0]

# temp_grid3 = data.x.numpy().squeeze().reshape(len(x_values), len(y_values))
# temp_grid4 = data.y.numpy().squeeze().reshape(len(x_values), len(y_values))

# fig, axs = plt.subplots(1, 4, figsize=(16, 4))
# # set font size
# plt.rcParams.update({'font.size': 18})

# axs[0].contourf(temp_grid1, levels=np.linspace(0, 1, 100), cmap='jet')
# axs[0].set_title('Resolution 8×8', y=-0.2)
# axs[0].axis('off')
# axs[1].contourf(temp_grid2, levels=np.linspace(0, 1, 100), cmap='jet')
# axs[1].set_title('Resolution 16×16', y=-0.2)
# axs[1].axis('off')
# axs[2].contourf(temp_grid3, levels=np.linspace(0, 1, 100), cmap='jet')
# axs[2].set_title('Resolution 32×32', y=-0.2)
# axs[2].axis('off')
# axs[3].contourf(temp_grid4, levels=np.linspace(0, 1, 100), cmap='jet')
# axs[3].set_title('Resolution 64×64', y=-0.2)
# axs[3].axis('off')
# # add colorbar on the right
# fig.tight_layout(pad=1.0)
# fig.colorbar(axs[0].contourf(temp_grid1, levels=np.linspace(0, 1, 100), cmap='jet'), ax=axs, location='right', label='Temperature (normalized)')

# plt.savefig('plots/figures/dataset_character_heat_trim.png')


# # plot burgers solutions for all four resolutions
# if os.path.exists(os.path.join("dataset/burger_plots", "processed")):
#     shutil.rmtree(os.path.join("dataset/burger_plots", "processed"))

# dataset = initialize_dataset(dataset="BurgersDataset", root="dataset/burger_plots", res_low=0, res_high=1, pre_transform='interpolate_high')
# data = dataset[0]
# pos_x = data.pos[:, 0]
# pos_y = data.pos[:, 1]

# x = data.x.numpy().squeeze()
# y = data.y.numpy().squeeze()

# x_values = np.unique(pos_x)
# y_values = np.unique(pos_y)

# temp_grid1 = x.reshape(len(x_values), len(y_values))
# temp_grid2 = y.reshape(len(x_values), len(y_values))

# if os.path.exists(os.path.join("dataset/burger_plots", "processed")):
#     shutil.rmtree(os.path.join("dataset/burger_plots", "processed"))

# dataset = initialize_dataset(dataset="BurgersDataset", root="dataset/burger_plots", res_low=2, res_high=3, pre_transform='interpolate_high')
# data = dataset[0]

# temp_grid3 = data.x.numpy().squeeze().reshape(len(x_values), len(y_values))
# temp_grid4 = data.y.numpy().squeeze().reshape(len(x_values), len(y_values))

# fig, axs = plt.subplots(1, 4, figsize=(16, 4))
# # set font size
# plt.rcParams.update({'font.size': 18})
# axs[0].contourf(temp_grid1, levels=np.linspace(0, 1, 100), cmap='jet')
# axs[0].set_title('Resolution 10×10', y=-0.2)
# axs[0].axis('off')

# axs[1].contourf(temp_grid2, levels=np.linspace(0, 1, 100), cmap='jet')
# axs[1].set_title('Resolution 20×20', y=-0.2)
# axs[1].axis('off')

# axs[2].contourf(temp_grid3, levels=np.linspace(0, 1, 100), cmap='jet')
# axs[2].set_title('Resolution 40×40', y=-0.2)
# axs[2].axis('off')

# axs[3].contourf(temp_grid4, levels=np.linspace(0, 1, 100), cmap='jet')
# axs[3].set_title('Resolution 80×80', y=-0.2)
# axs[3].axis('off')

# # add colorbar on the right
# fig.tight_layout(pad=1.0)
# fig.colorbar(axs[0].contourf(temp_grid1, levels=np.linspace(0, 1, 100), cmap='jet'), ax=axs, location='right', label='Velocity (normalized)')

# plt.savefig('plots/figures/dataset_character_burger_trim.png')

# # plot [1, 3] heat predictions of all models
# res_low = 1
# res_high = 3

# # load dataset
# dataset_dir = "dataset/heat"
# # dataset = initialize_dataset(dataset="HeatTransferDataset", root=dataset_dir, res_low=res_low, res_high=res_high, pre_transform='interpolate_high')
# # data = dataset[0]

# # load all models
# model_dir = "checkpoints/exp_1_heat"

# def inference(data, model_dir, res_low, res_high):
#     model_teecnet_dir = os.path.join(model_dir, "TEECNet", "res_{}_{}".format(res_low, res_high))
#     model_teecnet = initialize_model(type="TEECNet", in_channel=1, out_channel=1, width=16, num_layers=3, retrieve_weight=False)
#     model_teecnet.load_state_dict(torch.load(os.path.join(model_teecnet_dir, "model.pt")))
#     model_teecnet.eval()

#     t1 = time.time()
#     pred_teecnet = model_teecnet(data.x, data.edge_index, data.edge_attr)
#     t2 = time.time()
#     print("TEECNet inference time: {:.4f}s".format(t2 - t1))

#     del model_teecnet

#     model_graphsage_dir = os.path.join(model_dir, "GraphSAGE", "res_{}_{}".format(res_low, res_high))
#     model_graphsage = initialize_model(type="GraphSAGE", in_channel=1, out_channel=1, width=16, num_layers=6)
#     model_graphsage.load_state_dict(torch.load(os.path.join(model_graphsage_dir, "model.pt")))
#     model_graphsage.eval()

#     t1 = time.time()
#     pred_graphsage = model_graphsage(data.x, data.edge_index)
#     t2 = time.time()
#     print("GraphSAGE inference time: {:.4f}s".format(t2 - t1))

#     del model_graphsage

#     model_neuralope_dir = os.path.join(model_dir, "NeuralOperator", "res_{}_{}".format(res_low, res_high))
#     model_neuralope = initialize_model(type="NeuralOperator", in_channel=1, out_channel=1, width=64, num_layers=6)
#     model_neuralope.load_state_dict(torch.load(os.path.join(model_neuralope_dir, "model.pt")))
#     model_neuralope.eval()

#     t1 = time.time()
#     pred_neuralope = model_neuralope(data.x, data.edge_index, data.edge_attr)
#     t2 = time.time()
#     print("NeuralOperator inference time: {:.4f}s".format(t2 - t1))

#     del model_neuralope

#     return pred_teecnet, pred_graphsage, pred_neuralope

def inference_teecnet(data, model_dir, res_low, res_high):
    res_low = 0
    model_teecnet_dir = os.path.join(model_dir, "TEECNet", "res_{}_{}".format(res_low, res_high))
    model_teecnet = initialize_model(type="TEECNet", in_channel=2, out_channel=2, width=16, num_layers=3, retrieve_weight=False)
    model_teecnet.load_state_dict(torch.load(os.path.join(model_teecnet_dir, "model_130.pt")))
    model_teecnet.eval()

    t1 = time.time()
    pred_teecnet = model_teecnet(data.x, data.edge_index, data.edge_attr)
    t2 = time.time()
    print("TEECNet inference time: {:.4f}s".format(t2 - t1))

    del model_teecnet

    return pred_teecnet

def inference_graphsage(data, model_dir, res_low, res_high):
    res_low = 1
    model_graphsage_dir = os.path.join(model_dir, "GraphSAGE", "res_{}_{}".format(res_low, res_high))
    model_graphsage = initialize_model(type="GraphSAGE", in_channel=1, out_channel=1, width=16, num_layers=6)
    model_graphsage.load_state_dict(torch.load(os.path.join(model_graphsage_dir, "model.pt")))
    model_graphsage.eval()

    t1 = time.time()
    pred_graphsage = model_graphsage(data.x, data.edge_index)
    t2 = time.time()
    print("GraphSAGE inference time: {:.4f}s".format(t2 - t1))

    del model_graphsage

    return pred_graphsage

def inference_neuralope(data, model_dir, res_low, res_high):
    res_low = 1
    model_neuralope_dir = os.path.join(model_dir, "NeuralOperator", "res_{}_{}".format(res_low, res_high))
    model_neuralope = initialize_model(type="NeuralOperator", in_channel=1, out_channel=1, width=64, num_layers=6)
    model_neuralope.load_state_dict(torch.load(os.path.join(model_neuralope_dir, "model.pt")))
    model_neuralope.eval()

    t1 = time.time()
    pred_neuralope = model_neuralope(data.x, data.edge_index, data.edge_attr)
    t2 = time.time()
    print("NeuralOperator inference time: {:.4f}s".format(t2 - t1))

    del model_neuralope

    return pred_neuralope

# # select four samples for visualization
# sample_idx = [659, 800, 900, 969]
# fig, axs = plt.subplots(4, 5, figsize=(15.5, 10))
# x = 0
# # set font size
# plt.rcParams.update({'font.size': 16})
# for i in range(4):
#     dataset = initialize_dataset(dataset="HeatTransferDataset", root=dataset_dir, res_low=res_low, res_high=res_high, pre_transform='interpolate_high')
#     data = dataset[sample_idx[i]]
#     # release dataset memory
#     del dataset
#     pred_teecnet, pred_graphsage, pred_neuralope = inference(data, model_dir, res_low, res_high)

#     # plot
#     pos_x = data.pos[:, 0]
#     pos_y = data.pos[:, 1]

#     x = data.x.numpy().squeeze()
#     y = data.y.numpy().squeeze()

#     x_values = np.unique(pos_x)
#     y_values = np.unique(pos_y)

#     temp_grid1 = x.reshape(len(x_values), len(y_values))
#     temp_grid2 = y.reshape(len(x_values), len(y_values))

#     temp_grid3 = pred_teecnet.detach().numpy().squeeze().reshape(len(x_values), len(y_values))
#     temp_grid4 = pred_graphsage.detach().numpy().squeeze().reshape(len(x_values), len(y_values))
#     temp_grid5 = pred_neuralope.detach().numpy().squeeze().reshape(len(x_values), len(y_values))

#     axs[i, 0].contourf(temp_grid1, levels=np.linspace(0, 1, 100), cmap='jet')
#     if i == 0:
#         axs[i, 0].set_title('16×16 solution')
#     axs[i, 0].axis('off')

#     axs[i, 1].contourf(temp_grid3, levels=np.linspace(0, 1, 100), cmap='jet')
#     if i == 0:
#         axs[i, 1].set_title('TEECNet')
#     axs[i, 1].axis('off')

#     axs[i, 2].contourf(temp_grid4, levels=np.linspace(0, 1, 100), cmap='jet')
#     if i == 0:
#         axs[i, 2].set_title('GraphSAGE')
#     axs[i, 2].axis('off')

#     axs[i, 3].contourf(temp_grid5, levels=np.linspace(0, 1, 100), cmap='jet')
#     if i == 0:
#         axs[i, 3].set_title('Neural operator')
#     axs[i, 3].axis('off')

#     axs[i, 4].contourf(temp_grid2, levels=np.linspace(0, 1, 100), cmap='jet')
#     if i == 0:
#         axs[i, 4].set_title('64×64 solution')
#     axs[i, 4].axis('off')

#     # # add colorbar on the right of each row
#     # fig.colorbar(axs[i, 0].contourf(temp_grid1, levels=np.linspace(0, 1, 100), cmap='jet'), ax=axs[i, :], location='right')
#     del data, pred_teecnet, pred_graphsage, pred_neuralope
# # add colorbar on the right
# fig.tight_layout(pad=1.0)
# fig.colorbar(axs[i, 0].contourf(temp_grid1, levels=np.linspace(0, 1, 100), cmap='jet'), ax=axs, location='right', label='Temperature (normalized)',  pad=0.05)
# plt.savefig('plots/figures/inference_heat_trim.png')

# # plot [0, 3] burgers predictions of all models
# res_low = 0
# res_high = 3

# # load dataset
# dataset_dir = "dataset/burger"
# # data = dataset[0]

# # load all models
# model_dir = "checkpoints/exp_1_burger"

# sample_idx = [1, 4, 7, 16]

# fig, axs = plt.subplots(1, 5, figsize=(15.5, 6))
# # set font size
# plt.rcParams.update({'font.size': 16})
# for i in range(1):
#     dataset = initialize_dataset(dataset="BurgersDataset", root=dataset_dir, res_low=res_low, res_high=res_high, pre_transform='interpolate_high')

#     data = dataset[sample_idx[i]]
#     del dataset
#     # pred_teecnet, pred_graphsage, pred_neuralope = inference(data, model_dir, res_low, res_high)

#     # plot
#     pos_x = data.pos[:, 0]
#     pos_y = data.pos[:, 1]

#     # pos_x_low = data.pos_low[:, 0]
#     # pos_y_low = data.pos_low[:, 1]

#     x = data.x.numpy()
#     y = data.y.numpy()
#     x_low_x = x[:, 0]
#     x_low_y = x[:, 1]
#     x_low = np.sqrt(np.square(x_low_x) + np.square(x_low_y))
#     y_x = y[:, 0]
#     y_y = y[:, 1]
#     y = np.sqrt(np.square(y_x) + np.square(y_y))
#     # del data

#     x_values = np.unique(pos_x)
#     y_values = np.unique(pos_y)
#     # x_values_low = np.unique(pos_x_low)
#     # y_values_low = np.unique(pos_y_low)

#     del pos_x, pos_y

#     # temp_grid1 = x.reshape(len(x_values), len(y_values))
#     temp_grid1 = x_low.reshape(len(x_values), len(y_values))
#     temp_grid2 = y.reshape(len(x_values), len(y_values))

#     axs[0].contourf(temp_grid1, levels=100, cmap='jet')
#     if i == 0:
#         axs[0].set_title('10×10 solution')
#     axs[0].axis('off')

#     axs[4].contourf(temp_grid2, levels=100, cmap='jet')
#     if i == 0:
#         axs[4].set_title('80×80 solution')
#     # axs[i, 4].set_title('(e) 80×80 ground truth', y=-0.1)
#     axs[4].axis('off')

#     print("figure 1 5 done.")

#     del temp_grid2

#     pred_teecnet = inference_teecnet(data, model_dir, res_low, res_high)

#     # temp_grid3 = pred_teecnet.detach().numpy().squeeze().reshape(len(x_values), len(y_values))
#     temp_grid3 = pred_teecnet.detach().numpy()
#     temp_grid3_x = temp_grid3[:, 0].reshape(len(x_values), len(y_values))
#     temp_grid3_y = temp_grid3[:, 1].reshape(len(x_values), len(y_values))
#     temp_grid3 = np.sqrt(np.square(temp_grid3_x) + np.square(temp_grid3_y))
#     axs[1].contourf(temp_grid3, levels=100, cmap='jet')
#     if i == 0:
#         axs[1].set_title('TEECNet')
#     # axs[i, 1].set_title('(b) TEECNet', y=-0.1)
#     axs[1].axis('off')
#     del pred_teecnet, temp_grid3

#     # pred_graphsage = inference_graphsage(data, model_dir, res_low, res_high)
#     # temp_grid4 = pred_graphsage.detach().numpy().squeeze().reshape(len(x_values), len(y_values))
#     # axs[i, 2].contourf(temp_grid4, levels=np.linspace(0, 1, 100), cmap='jet')
#     # if i == 0:
#     #     axs[i, 2].set_title('GraphSAGE')
#     # # axs[i, 2].set_title('(c) GraphSAGE', y=-0.1)
#     # axs[i, 2].axis('off')

#     # del pred_graphsage, temp_grid4

#     # pred_neuralope = inference_neuralope(data, model_dir, res_low, res_high)
#     # temp_grid5 = pred_neuralope.detach().numpy().squeeze().reshape(len(x_values), len(y_values))
#     # axs[i, 3].contourf(temp_grid5, levels=np.linspace(0, 1, 100), cmap='jet')
#     # if i == 0:
#     #     axs[i, 3].set_title('Neural operator')
#     # # axs[i, 3].set_title('(d) Neural operator', y=-0.1)
#     # axs[i, 3].axis('off')

#     # del pred_neuralope, temp_grid5
#     # add colorbar on the right of each row
#     # fig.colorbar(axs[i, 0].contourf(temp_grid1, levels=np.linspace(0, 1, 100), cmap='jet'), ax=axs[i, :], location='right')

#     print("Sample {} done.".format(i))

# # add colorbar on the right
# fig.tight_layout(pad=1.0)
# fig.colorbar(axs[0].contourf(temp_grid1, levels=100, cmap='jet'), ax=axs, location='right', label='Velocity (normalized)',  pad=0.05)

# plt.savefig('plots/figures/inference_burger_trim.png')

# # plot inference time bar chart for three models
# res_low = 1
# res_high = 3

# # load dataset
# dataset_dir = "dataset/heat"
# dataset = initialize_dataset(dataset="HeatTransferDataset", root=dataset_dir, res_low=res_low, res_high=res_high, pre_transform='interpolate_high')
# # data = dataset[0]

# # load all models
# model_dir = "checkpoints/exp_1_heat"

# def inference(data, model_dir, res_low, res_high):
#     model_teecnet_dir = os.path.join(model_dir, "TEECNet", "res_{}_{}".format(res_low, res_high))
#     model_teecnet = initialize_model(type="TEECNet", in_channel=1, out_channel=1, width=16, num_layers=3, retrieve_weight=False)
#     model_teecnet.load_state_dict(torch.load(os.path.join(model_teecnet_dir, "model.pt")))
#     model_teecnet.eval()

#     t1 = time.time()
#     pred_teecnet = model_teecnet(data.x, data.edge_index, data.edge_attr)
#     t2 = time.time()
#     t_teecnet = t2 - t1

#     del model_teecnet

#     model_graphsage_dir = os.path.join(model_dir, "GraphSAGE", "res_{}_{}".format(res_low, res_high))
#     model_graphsage = initialize_model(type="GraphSAGE", in_channel=1, out_channel=1, width=16, num_layers=6)
#     model_graphsage.load_state_dict(torch.load(os.path.join(model_graphsage_dir, "model.pt")))
#     model_graphsage.eval()

#     t1 = time.time()
#     pred_graphsage = model_graphsage(data.x, data.edge_index)
#     t2 = time.time()
#     t_graphsage = t2 - t1

#     del model_graphsage

#     model_neuralope_dir = os.path.join(model_dir, "NeuralOperator", "res_{}_{}".format(res_low, res_high))
#     model_neuralope = initialize_model(type="NeuralOperator", in_channel=1, out_channel=1, width=64, num_layers=6)
#     model_neuralope.load_state_dict(torch.load(os.path.join(model_neuralope_dir, "model.pt")))
#     model_neuralope.eval()

#     t1 = time.time()
#     pred_neuralope = model_neuralope(data.x, data.edge_index, data.edge_attr)
#     t2 = time.time()
#     t_neuralope = t2 - t1

#     del model_neuralope

#     return t_teecnet, t_graphsage, t_neuralope

# T_teecnet = []
# T_graphsage = []
# T_neuralope = []

# for data in dataset:
#     t_teecnet, t_graphsage, t_neuralope = inference(data, model_dir, res_low, res_high)
#     T_teecnet.append(t_teecnet)
#     T_graphsage.append(t_graphsage)
#     T_neuralope.append(t_neuralope)

# T_teecnet = np.array(T_teecnet)
# T_graphsage = np.array(T_graphsage)
# T_neuralope = np.array(T_neuralope)

# # plot
# fig, ax = plt.subplots(figsize=(8, 6))
# # set font size
# plt.rcParams.update({'font.size': 18})
# ax.bar([1, 2, 3], [np.mean(T_teecnet), np.mean(T_graphsage), np.mean(T_neuralope)], yerr=[np.std(T_teecnet), np.std(T_graphsage), np.std(T_neuralope)], capsize=10)
# ax.set_xticks([1, 2, 3])
# ax.set_xticklabels(['TEECNet', 'GraphSAGE', 'Neural operator'], size=18)
# ax.set_ylabel('Inference time (s)', size=18)
# plt.savefig('plots/figures/inference_time_heat.png')

# # calculate percentage of time reduction between TEECNet and Neural Operator
# print("TEECNet inference time: {:.4f}+-{:.4f}s".format(np.mean(T_teecnet), np.std(T_teecnet)))
# print("NeuralOperator inference time: {:.4f}s+-{:.4f}s".format(np.mean(T_neuralope), np.std(T_neuralope)))

# print("Percentage of time reduction: {:.2f}%".format((np.mean(T_neuralope) - np.mean(T_teecnet)) / np.mean(T_neuralope) * 100))

# plot geometrical variance heat predictions

# res_low = 1
# res_high = 3

# # load dataset
# dataset_dir = "dataset/heat_geometry_test"
# dataset = initialize_dataset(dataset="HeatTransferMultiGeometryDataset", root=dataset_dir, res_low=res_low, res_high=res_high, pre_transform='interpolate_high')

# # load all models
# model_dir = "checkpoints/exp_1_heat"

# def inference(data, model_dir, res_low, res_high):
#     model_teecnet_dir = os.path.join(model_dir, "TEECNet", "res_{}_{}".format(res_low, res_high))
#     model_teecnet = initialize_model(type="TEECNet", in_channel=1, out_channel=1, width=16, num_layers=3, retrieve_weight=False)
#     model_teecnet.load_state_dict(torch.load(os.path.join(model_teecnet_dir, "model.pt")))
#     model_teecnet.eval()

#     t1 = time.time()
#     pred_teecnet = model_teecnet(data.x, data.edge_index, data.edge_attr)
#     t2 = time.time()

#     del model_teecnet

#     model_graphsage_dir = os.path.join(model_dir, "GraphSAGE", "res_{}_{}".format(res_low, res_high))
#     model_graphsage = initialize_model(type="GraphSAGE", in_channel=1, out_channel=1, width=16, num_layers=6)
#     model_graphsage.load_state_dict(torch.load(os.path.join(model_graphsage_dir, "model.pt")))
#     model_graphsage.eval()

#     t1 = time.time()
#     pred_graphsage = model_graphsage(data.x, data.edge_index)
#     t2 = time.time()

#     del model_graphsage

#     model_neuralope_dir = os.path.join(model_dir, "NeuralOperator", "res_{}_{}".format(res_low, res_high))
#     model_neuralope = initialize_model(type="NeuralOperator", in_channel=1, out_channel=1, width=64, num_layers=6)
#     model_neuralope.load_state_dict(torch.load(os.path.join(model_neuralope_dir, "model.pt")))
#     model_neuralope.eval()

#     t1 = time.time()
#     pred_neuralope = model_neuralope(data.x, data.edge_index, data.edge_attr)
#     t2 = time.time()

#     del model_neuralope

#     return pred_teecnet, pred_graphsage, pred_neuralope

# fig, axs = plt.subplots(4, 5, figsize=(15.5, 10))
# # set font size
# plt.rcParams.update({'font.size': 16})
# data_samples = [2, 5, 8, 9]
# for i in range(4):
#     data = dataset[data_samples[i]]
#     pred_teecnet, pred_graphsage, pred_neuralope = inference(data, model_dir, res_low, res_high)

#     # plot
#     pos_x = data.pos[:, 0]
#     pos_y = data.pos[:, 1]

#     tri = Triangulation(pos_x, pos_y, data.cells.numpy())
#     axs[i, 0].tricontourf(tri, data.x.numpy().squeeze(), levels=np.linspace(0, 1, 100), cmap='jet')
#     if i == 0:
#         axs[i, 0].set_title('0.05 solution')
#     axs[i, 0].axis('off')

#     axs[i, 4].tricontourf(tri, data.y.numpy().squeeze(), levels=np.linspace(0, 1, 100), cmap='jet')
#     if i == 0:
#         axs[i, 4].set_title('0.0125 solution')
#     axs[i, 4].axis('off')

#     pred_teecnet = pred_teecnet.detach().numpy().squeeze()
#     axs[i, 1].tricontourf(tri, pred_teecnet, levels=np.linspace(0, 1, 100), cmap='jet')
#     if i == 0:
#         axs[i, 1].set_title('TEECNet')
#     axs[i, 1].axis('off')

#     pred_graphsage = pred_graphsage.detach().numpy().squeeze()
#     axs[i, 2].tricontourf(tri, pred_graphsage, levels=np.linspace(0, 1, 100), cmap='jet')
#     if i == 0:
#         axs[i, 2].set_title('GraphSAGE')
#     axs[i, 2].axis('off')

#     pred_neuralope = pred_neuralope.detach().numpy().squeeze()
#     axs[i, 3].tricontourf(tri, pred_neuralope, levels=np.linspace(0, 1, 100), cmap='jet')
#     if i == 0:
#         axs[i, 3].set_title('Neural operator')
#     axs[i, 3].axis('off')

#     del pred_teecnet, pred_graphsage, pred_neuralope

# # add colorbar on the right of each row
# fig.tight_layout(pad=1.0)
# fig.colorbar(axs[i, 0].tricontourf(tri, data.x.numpy().squeeze(), levels=np.linspace(0, 1, 100), cmap='jet'), ax=axs, location='right', label='Temperature (normalized)', pad=0.05)

# plt.savefig('plots/figures/inference_heat_geometry.png')

# # plot three variant heat data samples
# res_low = 1
# res_high = 3

# # load dataset
# dataset_dir = "dataset/heat"
# dataset = initialize_dataset(dataset="HeatTransferDataset", root=dataset_dir, res_low=res_low, res_high=res_high, pre_transform='interpolate_high')


# data_samples = [[100, 101, 102], [104, 105, 106]]
# for i in range(2):
#     fig, axs = plt.subplots(1, 3, figsize=(18, 5))
#     for j in range(3):
#         data = dataset[data_samples[i][j]]

#         # plot
#         pos_x = data.pos[:, 0]
#         pos_y = data.pos[:, 1]

#         x_values = np.unique(pos_x)
#         y_values = np.unique(pos_y)

#         temp_grid = data.y.numpy().squeeze().reshape(len(x_values), len(y_values))

#         axs[j].contourf(temp_grid, levels=np.linspace(0, 1, 100), cmap='jet')
    
#         axs[j].axis('off')

#         del data

#     plt.savefig('plots/figures/heat_data_samples_{}.png'.format(i))

# # plot input, prediction, ground truth and error map of burgers dataset
# res_low = 0
# res_high = 3

# # load dataset
# dataset_dir = "dataset/burger_plots"
# dataset = initialize_dataset(dataset="BurgersDataset", root=dataset_dir, res_low=res_low, res_high=res_high, pre_transform='interpolate_high')

# # load all models
# model_dir = "checkpoints/exp_1_burger"
# fig, axs = plt.subplots(4, 4, figsize=(13, 10))
# data_idx = [1, 4, 7, 16]
# for i in range(4):
#     data = dataset[data_idx[i]]
#     pred_teecnet = inference_teecnet(data, model_dir, res_low, res_high)

#     # plot
#     pos_x = data.pos[:, 0]
#     pos_y = data.pos[:, 1]

#     pos_x_low = data.pos_low[:, 0]
#     pos_y_low = data.pos_low[:, 1]

#     x = data.x.numpy().squeeze()
#     y = data.y.numpy().squeeze()
#     x_low = data.x_low.numpy().squeeze()

#     x_values = np.unique(pos_x)
#     y_values = np.unique(pos_y)

#     x_values_low = np.unique(pos_x_low)
#     y_values_low = np.unique(pos_y_low)

#     temp_grid1 = x.reshape(len(x_values), len(y_values))
#     temp_grid2 = y.reshape(len(x_values), len(y_values))

#     temp_grid3 = pred_teecnet.detach().numpy().squeeze().reshape(len(x_values), len(y_values))

#     temp_grid4 = temp_grid2 - temp_grid3

#     temp_grid_low = x_low.reshape(len(x_values_low), len(y_values_low))

#     axs[i, 0].contourf(temp_grid_low, levels=np.linspace(0, 1, 100), cmap='jet')
#     if i == 0:
#         axs[i, 0].set_title('(a) 10×10 solution')
#     axs[i, 0].axis('off')

#     axs[i, 1].contourf(temp_grid3, levels=np.linspace(0, 1, 100), cmap='jet')
#     if i == 0:
#         axs[i, 1].set_title('(b) TEECNet')
#     axs[i, 1].axis('off')

#     axs[i, 2].contourf(temp_grid2, levels=np.linspace(0, 1, 100), cmap='jet')
#     if i == 0:
#         axs[i, 2].set_title('(c) 80×80 solution')
#     axs[i, 2].axis('off')

#     axs[i, 3].contourf(temp_grid4, levels=np.linspace(-0.1, 1, 100), cmap='jet')
#     if i == 0:
#         axs[i, 3].set_title('(d) Error map')
#     axs[i, 3].axis('off')

#     del data, pred_teecnet

# # add colorbar on the right
# fig.tight_layout(pad=1.0)
# fig.colorbar(axs[i, 0].contourf(temp_grid1, levels=np.linspace(0, 1, 100), cmap='jet'), ax=axs, location='right', label='Velocity (normalized)',  pad=0.05)
# plt.savefig('plots/figures/inference_burger_error.png')


# # plot mesh geometry and color the nodes with the heatmap of the solution at that node
# res_low = 1
# res_high = 3

# dataset_dir = "dataset/heat_geometry_test"

# dataset = initialize_dataset(dataset="HeatTransferMultiGeometryDataset", root=dataset_dir, res_low=res_low, res_high=res_high, pre_transform='interpolate_high')

# data_sample = dataset[0]

# x = data_sample.pos[:, 0].numpy()
# y = data_sample.pos[:, 1].numpy()

# T = data_sample.y.numpy().squeeze()
# cells = data_sample.cells.numpy()

# edge_length = data_sample.edge_attr[:, 0].numpy()

# tri = Triangulation(x, y, cells)
# cmap = plt.cm.jet
# fig, ax = plt.subplots(figsize=(20, 20))
# # plot the mesh geometry with edges colored by edge length
# ax.triplot(tri, color='black', linewidth=0.3)
# # ax.tripcolor(tri, edge_length, cmap=cmap)
# # plot the mesh nodes with color representing the solution
# ax.scatter(x, y, c=T, cmap=cmap, s=0.7)
# # save vector figure
# plt.savefig('plots/figures/heat_geometry.eps', format='eps')


# # plot mesh geometry and color the edges with the computed weight of the edge in TEECNet
# res_low = 1
# res_high = 3

# dataset_dir = "dataset/heat_geometry_test"

# dataset = initialize_dataset(dataset="HeatTransferMultiGeometryDataset", root=dataset_dir, res_low=res_low, res_high=res_high, pre_transform='interpolate_high')

# data_sample = dataset[0]

# x = data_sample.pos[:, 0].numpy()
# y = data_sample.pos[:, 1].numpy()

# model_dir = "checkpoints/exp_1_heat"
# model_teecnet_dir = os.path.join(model_dir, "TEECNet", "res_0_{}".format(res_low, res_high))
# model_teecnet = initialize_model(type="TEECNet", in_channel=1, out_channel=1, width=16, num_layers=3, retrieve_weight=True)
# model_teecnet.load_state_dict(torch.load(os.path.join(model_teecnet_dir, "model_190.pt")))
# model_teecnet.eval()

# pred_teecnet = model_teecnet(data_sample.x, data_sample.edge_index, data_sample.edge_attr)

# edge_weight_k = model_teecnet.kernel.weight_k.detach().numpy()
# edge_weight_op = model_teecnet.kernel.weight_op.detach().numpy()

# edge_weight_op = np.random.rand(edge_weight_op.shape[0], edge_weight_op.shape[1], edge_weight_op.shape[2])
# edge_weights = edge_weight_k + edge_weight_op
# edge_weights = np.sum(edge_weights, axis=1)
# edge_weights = np.sum(edge_weights, axis=1)
# print(max(edge_weights))
# print(min(edge_weights))

# cells = data_sample.cells.numpy()

# edge_length = data_sample.edge_attr[:, 0].numpy()

# tri = Triangulation(x, y, cells)

# cmap = plt.cm.jet
# fig, ax = plt.subplots(figsize=(20, 20))
# # plot the mesh geometry with edges colored by edge weight
# # ax.triplot(tri, color='black', linewidth=0.3)
# line_lists = tri.edges
# lines = []
# for i in range(line_lists.shape[0]):
#     lines.append([(x[line_lists[i, 0]], y[line_lists[i, 0]]), (x[line_lists[i, 1]], y[line_lists[i, 1]])])

# lines = LineCollection(lines, cmap=cmap)
# lines.set_array(edge_weights)
# lines.set_linewidth(0.5)

# ax.add_collection(lines)

# # plot the points
# ax.scatter(x, y, c='black', s=0.7)
# # save vector figure
# plt.savefig('plots/figures/heat_geometry_weight.eps', format='eps')


# plot megaflow inference results
dataset_dir = "dataset/reduced_megaflow"
dataset = initialize_dataset(dataset="MegaFlow2D", root=dataset_dir, pre_transform='interpolate_high')

model = initialize_model(type="TEECNet", in_channel=3, out_channel=3, width=16, num_layers=3, retrieve_weight=False, num_powers=2)
# model = initialize_model(type="NeuralOperator", in_channel=3, out_channel=3, width=64, num_layers=4)
model_dir = "checkpoints/exp_1_megaflow/TEECNet/res_0_1/model_90.pt"
# model_dir = "checkpoints/exp_1_megaflow/NeuralOperator/res_0_1/model_80.pt"
model.load_state_dict(torch.load(model_dir))
model.eval()

data = dataset[800]

pred = model(data.x, data.edge_index, data.edge_attr)

velocity_low = data.x[:, 0:2].numpy()
velocity_high = data.y[:, 0:2].numpy()
velocity_pred = pred[:, 0:2].detach().numpy()

velocity_low = np.sqrt(np.square(velocity_low[:, 0]) + np.square(velocity_low[:, 1]))
velocity_high = np.sqrt(np.square(velocity_high[:, 0]) + np.square(velocity_high[:, 1]))
velocity_pred = np.sqrt(np.square(velocity_pred[:, 0]) + np.square(velocity_pred[:, 1]))

# plot
pos_x = data.pos[:, 0].numpy()
pos_y = data.pos[:, 1].numpy()

tri = Triangulation(pos_x, pos_y, data.cells.numpy())

fig, axs = plt.subplots(1, 3, figsize=(48, 5))
# set font size
plt.rcParams.update({'font.size': 16})

# axs[0].tricontourf(tri, velocity_low, levels=100, cmap='jet')
fig.colorbar(axs[0].tricontourf(tri, velocity_low, levels=np.linspace(0, 5, 100), cmap='jet'), ax=axs, location='right', label='Velocity (normalized)',  pad=0.05)
# axs[1].tricontourf(tri, velocity_pred, levels=100, cmap='jet')
fig.colorbar(axs[1].tricontourf(tri, velocity_pred, levels=np.linspace(0, 5, 100), cmap='jet'), ax=axs, location='right', label='Velocity (normalized)',  pad=0.05)
# axs[2].tricontourf(tri, velocity_high, levels=100, cmap='jet')
fig.colorbar(axs[2].tricontourf(tri, velocity_high, levels=np.linspace(0, 5, 100), cmap='jet'), ax=axs, location='right', label='Velocity (normalized)',  pad=0.05)

axs[0].set_title('LAS solution')
axs[1].set_title('TEECNet')
# axs[1].set_title('Neural operator')
axs[2].set_title('HAS solution')

axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')

# add colorbar on the right
plt.tight_layout(pad=1.0)
fig.colorbar(axs[0].tricontourf(tri, velocity_low, levels=100, cmap='jet'), ax=axs, location='right', label='Velocity (normalized)',  pad=0.05)
plt.savefig('plots/figures/inference_megaflow_trim.png')


