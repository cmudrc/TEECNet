import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import *


def save_prediction_results(flowMLdata, model, model_dir, save_path=None):
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

    flowMLdata = flowMLdata.to(device)
    has_star = model(flowMLdata)
    has_star = has_star.detach().numpy()
    has = flowMLdata.y.detach().numpy()
    las = flowMLdata.x.detach().numpy()

    u_has_star = [has_star[:, 0], has_star[:, 1]]
    p_has_star = has_star[:, 2]

    u_has = [has[:, 0], has[:, 1]]
    p_has = has[:, 2]

    u_las = [las[:, 0], las[:, 1]]
    p_las = las[:, 2]

    np.savez(save_path, u_has_star=u_has_star, p_has_star=p_has_star, u_has=u_has, p_has=p_has, u_las=u_las, p_las=p_las)


if __name__ == "__main__":
    train_config = load_yaml("config/train_config.yaml")
    flowMLmodel = initialize_model(in_channel=3, out_channel=3, layers=3, num_filters=[8, 16, 8], type="CFDError")
    flowMLdataset = initialize_dataset(dataset="MegaFlow2D", split_scheme=train_config["split_scheme"], dir=train_config["data_dir"], transform=train_config["transform"], split_ratio=train_config["split_ratio"], pre_transform=None)
    flowMLdata, data_name = flowMLdataset.get_eval(1205688)
    print(data_name)

    save_prediction_results(flowMLdata, flowMLmodel, model_dir="D:/Work/research/train/checkpoints/2023-01-18_10-53/checkpoint-000675.pth", save_path="D:/Work/research/train/prediction/prediction_result_{}_model_675.npz".format(data_name))