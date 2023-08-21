import os
import time
from utils import train_test_split, get_cur_time, initialize_model, initialize_dataset, parse_args, load_yaml
import matplotlib.pyplot as plt
import numpy as np
import shutil
import torch

# # plot heat solutions for all four resolutions
# if os.path.exists(os.path.join("dataset/heat", "processed")):
# shutil.rmtree(os.path.join("dataset/heat", "processed"))

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
# shutil.rmtree(os.path.join("dataset/heat", "processed"))

# dataset = initialize_dataset(dataset="HeatTransferDataset", root="dataset/heat", res_low=2, res_high=3, pre_transform='interpolate_high')
# data = dataset[0]

# temp_grid3 = data.x.numpy().squeeze().reshape(len(x_values), len(y_values))
# temp_grid4 = data.y.numpy().squeeze().reshape(len(x_values), len(y_values))

# fig, axs = plt.subplots(2, 2, figsize=(13, 10))
# axs[0, 0].contourf(temp_grid1, levels=np.linspace(0, 1, 100), cmap='jet')
# axs[0, 0].set_title('(a) Resolution 8×8', y=-0.1)
# axs[0, 0].axis('off')
# axs[0, 1].contourf(temp_grid2, levels=np.linspace(0, 1, 100), cmap='jet')
# axs[0, 1].set_title('(b) Resolution 16×16', y=-0.1)
# axs[0, 1].axis('off')
# axs[1, 0].contourf(temp_grid3, levels=np.linspace(0, 1, 100), cmap='jet')
# axs[1, 0].set_title('(c) Resolution 32×32', y=-0.1)
# axs[1, 0].axis('off')
# axs[1, 1].contourf(temp_grid4, levels=np.linspace(0, 1, 100), cmap='jet')
# axs[1, 1].set_title('(d) Resolution 64×64', y=-0.1)
# axs[1, 1].axis('off')
# # add colorbar on the right
# fig.colorbar(axs[0, 0].contourf(temp_grid1, levels=np.linspace(0, 1, 100), cmap='jet'), ax=axs, location='right')
# plt.savefig('plots/figures/dataset_character_heat.png')


# # plot burgers solutions for all four resolutions
# if os.path.exists(os.path.join("dataset/burger", "processed")):
#     shutil.rmtree(os.path.join("dataset/burger", "processed"))

# dataset = initialize_dataset(dataset="BurgersDataset", root="dataset/burger", res_low=0, res_high=1, pre_transform='interpolate_high')
# data = dataset[0]
# pos_x = data.pos[:, 0]
# pos_y = data.pos[:, 1]

# x = data.x.numpy().squeeze()
# y = data.y.numpy().squeeze()

# x_values = np.unique(pos_x)
# y_values = np.unique(pos_y)

# temp_grid1 = x.reshape(len(x_values), len(y_values))
# temp_grid2 = y.reshape(len(x_values), len(y_values))

# if os.path.exists(os.path.join("dataset/burger", "processed")):
#     shutil.rmtree(os.path.join("dataset/burger", "processed"))

# dataset = initialize_dataset(dataset="BurgersDataset", root="dataset/burger", res_low=2, res_high=3, pre_transform='interpolate_high')
# data = dataset[0]

# temp_grid3 = data.x.numpy().squeeze().reshape(len(x_values), len(y_values))
# temp_grid4 = data.y.numpy().squeeze().reshape(len(x_values), len(y_values))

# fig, axs = plt.subplots(2, 2, figsize=(13, 10))
# axs[0, 0].contourf(temp_grid1, levels=np.linspace(0, 1, 100), cmap='jet')
# axs[0, 0].set_title('(a) Resolution 10×10', y=-0.1)
# axs[0, 0].axis('off')

# axs[0, 1].contourf(temp_grid2, levels=np.linspace(0, 1, 100), cmap='jet')
# axs[0, 1].set_title('(b) Resolution 20×20', y=-0.1)
# axs[0, 1].axis('off')

# axs[1, 0].contourf(temp_grid3, levels=np.linspace(0, 1, 100), cmap='jet')
# axs[1, 0].set_title('(c) Resolution 40×40', y=-0.1)
# axs[1, 0].axis('off')

# axs[1, 1].contourf(temp_grid4, levels=np.linspace(0, 1, 100), cmap='jet')
# axs[1, 1].set_title('(d) Resolution 80×80', y=-0.1)
# axs[1, 1].axis('off')

# # add colorbar on the right
# fig.colorbar(axs[0, 0].contourf(temp_grid1, levels=np.linspace(0, 1, 100), cmap='jet'), ax=axs, location='right')

# plt.savefig('plots/figures/dataset_character_burger.png')

# plot [1, 3] predictions of all models
res_low = 1
res_high = 3

# load dataset
dataset_dir = "dataset/heat"
dataset = initialize_dataset(dataset="HeatTransferDataset", root=dataset_dir, res_low=res_low, res_high=res_high, pre_transform='interpolate_high')
# data = dataset[0]

# load all models
model_dir = "checkpoints/exp_1_heat"

def inference(data, model_dir, res_low, res_high):
    model_teecnet_dir = os.path.join(model_dir, "TEECNet", "checkpoints", "res_{}_{}".format(res_low, res_high))
    model_teecnet = initialize_model(model="TEECNet", in_channel=1, out_channel=1, width=16, num_layers=3)
    model_teecnet.load_state_dict(torch.load(os.path.join(model_teecnet_dir, "model.pth")))
    model_teecnet.eval()

    t1 = time.time()
    pred_teecnet = model_teecnet(data.x, data.edge_index, data.edge_attr)
    t2 = time.time()
    print("TEECNet inference time: {:.4f}s".format(t2 - t1))

    model_graphsage_dir = os.path.join(model_dir, "GraphSAGE", "checkpoints", "res_{}_{}".format(res_low, res_high))
    model_graphsage = initialize_model(model="GraphSAGE", in_channel=1, out_channel=1, width=16, num_layers=6)
    model_graphsage.load_state_dict(torch.load(os.path.join(model_graphsage_dir, "model.pth")))
    model_graphsage.eval()

    t1 = time.time()
    pred_graphsage = model_graphsage(data.x, data.edge_index)
    t2 = time.time()
    print("GraphSAGE inference time: {:.4f}s".format(t2 - t1))

    model_neuralope_dir = os.path.join(model_dir, "NeuralOperator", "checkpoints", "res_{}_{}".format(res_low, res_high))
    model_neuralope = initialize_model(model="NeuralOperator", in_channel=1, out_channel=1, width=512, num_layers=6)
    model_neuralope.load_state_dict(torch.load(os.path.join(model_neuralope_dir, "model.pth")))
    model_neuralope.eval()

    t1 = time.time()
    pred_neuralope = model_neuralope(data.x, data.edge_index)
    t2 = time.time()
    print("NeuralOperator inference time: {:.4f}s".format(t2 - t1))

    return pred_teecnet, pred_graphsage, pred_neuralope

# select four samples for visualization
sample_idx = [4000, 4500, 4600, 4700]
fig, axs = plt.subplots(5, 3, figsize=(13, 10))
for i in range(4):
    data = dataset[sample_idx[i]]
    pred_teecnet, pred_graphsage, pred_neuralope = inference(data, model_dir, res_low, res_high)

    # plot
    pos_x = data.pos[:, 0]
    pos_y = data.pos[:, 1]

    x = data.x.numpy().squeeze()
    y = data.y.numpy().squeeze()

    x_values = np.unique(pos_x)
    y_values = np.unique(pos_y)

    temp_grid1 = x.reshape(len(x_values), len(y_values))
    temp_grid2 = y.reshape(len(x_values), len(y_values))

    temp_grid3 = pred_teecnet.detach().numpy().squeeze().reshape(len(x_values), len(y_values))
    temp_grid4 = pred_graphsage.detach().numpy().squeeze().reshape(len(x_values), len(y_values))
    temp_grid5 = pred_neuralope.detach().numpy().squeeze().reshape(len(x_values), len(y_values))

    axs[i, 0].contourf(temp_grid1, levels=np.linspace(0, 1, 100), cmap='jet')
    axs[i, 0].set_title('(a) 16×16 input', y=-0.1)
    axs[i, 0].axis('off')

    axs[i, 1].contourf(temp_grid3, levels=np.linspace(0, 1, 100), cmap='jet')
    axs[i, 1].set_title('(b) TEECNet', y=-0.1)
    axs[i, 1].axis('off')

    axs[i, 2].contourf(temp_grid4, levels=np.linspace(0, 1, 100), cmap='jet')
    axs[i, 2].set_title('(c) GraphSAGE', y=-0.1)
    axs[i, 2].axis('off')

    axs[i, 3].contourf(temp_grid5, levels=np.linspace(0, 1, 100), cmap='jet')
    axs[i, 3].set_title('(d) Neural operator', y=-0.1)
    axs[i, 3].axis('off')

    axs[i, 4].contourf(temp_grid2, levels=np.linspace(0, 1, 100), cmap='jet')
    axs[i, 4].set_title('(e) 64×64 ground truth', y=-0.1)
    axs[i, 4].axis('off')

    # add colorbar on the right of each row
    fig.colorbar(axs[i, 0].contourf(temp_grid1, levels=np.linspace(0, 1, 100), cmap='jet'), ax=axs[i, :], location='right')

plt.savefig('plots/figures/inference_heat.png')
