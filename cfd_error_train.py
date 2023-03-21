import os
import numpy as np
from functools import partial
import torch
from torch_geometric.loader import DataLoader
from utils import *

from torch.utils.tensorboard import SummaryWriter


def train_cfderror(train_config, checkpoint_dir=None):
    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # setup model according to command line arguments
    model = initialize_model(in_channel=train_config["in_channel"], out_channel=train_config["out_channel"], type=train_config["model_name"], layers=None, num_filters=None)
    if checkpoint_dir:
        checkpoint_load(model, checkpoint_dir)
    
    model = model.to(device)
    print(model)

    # setup dataset
    dataset = initialize_dataset(dataset="MegaFlow2D", split_scheme=train_config["split_scheme"], dir=train_config["data_dir"], transform=train_config["transform"], split_ratio=train_config["split_ratio"], pre_transform=None)
    # dataset.process() # test dataset processing parallel
    print(dataset)
    # test_data_l, test_data_h = dataset.get(8000)
    # print(test_data_l)

    # split dataset into train, val and test sets
    train_dataset = dataset[:int(len(dataset) * 0.8)]
    val_dataset = dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)]
    test_dataset = dataset[int(len(dataset) * 0.9):]

    # setup dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True, num_workers=20)
    val_dataloader = DataLoader(val_dataset, batch_size=train_config["batch_size"], shuffle=False, num_workers=20)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    # setup loss function
    loss_fn = initialize_loss(loss_type=train_config["loss"])

    # setup metric function
    metric_fn = initialize_metric(metric_type=train_config["metric"])

    # setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["learning_rate"])

    # setup tensorboard
    logdir = '../train/logs/{}'.format(get_cur_time())
    savedir = '../train/checkpoints/{}'.format(get_cur_time())
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    writer_logs = SummaryWriter(logdir)

    for epoch in range(train_config["epochs"]):
        model.train()
        avg_loss = 0
        avg_accuracy = 0
        for batch in train_dataloader:
            batch_l, batch_h = batch[0], batch[1]
            batch_l, batch_h = batch_l.to(device), batch_h.to(device)
            optimizer.zero_grad()
            pred = model(batch_l, batch_h)
            loss = loss_fn.compute(batch_h.x, pred)
            avg_loss += loss.item()
            avg_accuracy += metric_fn.compute(batch_h.x, pred)
            loss.backward()
            optimizer.step()

        avg_loss /= len(train_dataloader)
        avg_accuracy /= len(train_dataloader)
        print('Epoch: {:03d}, Loss: {:.4f}, Accuracy metric: {:4f}'.format(epoch, avg_loss, avg_accuracy))

        writer_logs.add_scalar('Loss/train', avg_loss, epoch)
        writer_logs.add_scalar('Max_div/train', avg_accuracy, epoch)

        if epoch % 25 == 0:
            evaluate_model(model, val_dataloader, writer_logs, epoch, loss_fn, metric_fn, device, mode='val')
            checkpoint_save(model, savedir, epoch)

    evaluate_model(model, test_dataloader, writer_logs, epoch, loss_fn, metric_fn, device, mode='test')
    writer_logs.close()

if __name__ == '__main__':
    train_config = load_yaml("config/train_config.yaml")
    train_cfderror(train_config)