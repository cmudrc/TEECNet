import os
import numpy as np
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
    train_dataloader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=False, num_workers=10, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=train_config["batch_size"], shuffle=False, num_workers=10, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, collate_fn=collate_fn)

    # setup loss function
    loss_fn = initialize_loss(loss_type=train_config["loss"])

    # setup metric function
    metric_fn = initialize_metric(metric_type=train_config["metric"])

    # setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["learning_rate"], weight_decay=train_config["weight_decay"])

    # setup scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-6, eps=1e-08)

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
        for (batch_l, batch_h) in train_dataloader:
            batch_l, batch_h = batch_l.to(device), batch_h.to(device)
            optimizer.zero_grad()
            pred = model(batch_l, batch_h)
            # loss = loss_fn.compute(batch_h.x, pred)
            loss = loss_fn.compute(pred, batch_h.x, batch_h.pos, batch_h.edge_index, weight=0.001)
            avg_loss += loss.item()
            accuracy = metric_fn.compute(batch_h.x, pred).item()
            avg_accuracy += accuracy
            # print('Epoch: {:03d}, Batch: {:03d}, Loss: {:.4f}, Accuracy metric: {:4f}'.format(epoch, len(train_dataloader), loss.item(), accuracy))
            loss.backward()
            
            optimizer.step()

        avg_loss /= len(train_dataloader)
        avg_accuracy /= len(train_dataloader)
        print('Epoch: {:03d}, Loss: {:.4f}, Accuracy metric: {:4f}'.format(epoch, avg_loss, avg_accuracy))

        writer_logs.add_scalar('Loss/train', avg_loss, epoch)
        writer_logs.add_scalar('Max_div/train', avg_accuracy, epoch)

        if epoch % 10 == 0:
            val_loss, val_metric = evaluate_model(model, val_dataloader, writer_logs, epoch, loss_fn, metric_fn, device, mode='val')
            checkpoint_save(model, savedir, epoch)
            scheduler.step(val_loss)

    evaluate_model(model, test_dataloader, writer_logs, epoch, loss_fn, metric_fn, device, mode='test')
    writer_logs.close()

if __name__ == '__main__':
    train_config = load_yaml("config/train_config.yaml")
    train_cfderror(train_config, checkpoint_dir="D:/Work/research/train/checkpoints/2023-03-27_17-00/checkpoint-000020.pth")