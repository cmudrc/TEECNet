import os
import numpy as np
from functools import partial
import torch
from torch_geometric.loader import DataLoader
from utils import *
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from torch.utils.tensorboard import SummaryWriter


def train_cfderror(tune_config, train_config, checkpoint_dir=None, data_dir=None):
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
    dataset = initialize_dataset(dataset="MegaFlow2D", split_scheme=train_config["split_scheme"], dir=data_dir, transform=train_config["transform"], split_ratio=train_config["split_ratio"], pre_transform=None)
    # dataset.process() # test dataset processing parallel
    print(dataset)

    # split dataset into train, val and test sets
    train_dataset = dataset[:int(len(dataset) * 0.8)]
    val_dataset = dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)]
    test_dataset = dataset[int(len(dataset) * 0.9):]

    # setup dataloader
    train_dataloader = DataLoader(dataset, batch_size=tune_config["batch_size"], shuffle=True, num_workers=16)
    val_dataloader = DataLoader(val_dataset, batch_size=tune_config["batch_size"], shuffle=False, num_workers=16)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    # setup loss function
    loss_fn = initialize_loss(loss_type=train_config["loss"])

    # setup metric function
    metric_fn = initialize_metric(metric_type=train_config["metric"])

    # setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=tune_config["lr"])

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
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn.compute(batch.y, pred)
            avg_loss += loss.item()
            avg_accuracy += metric_fn.compute(batch.y, pred)
            loss.backward()
            optimizer.step()
            # print('Epoch: {:03d}, Batch: {:03d}, Loss: {:.4f}, Accuracy metric: {:4f}'.format(epoch, batch.batch[-1], loss.item(), metric_fn(batch.y, pred)))

        avg_loss /= len(train_dataloader)
        avg_accuracy /= len(train_dataloader)
        print('Epoch: {:03d}, Loss: {:.4f}, Accuracy metric: {:4f}'.format(epoch, avg_loss, avg_accuracy))

        writer_logs.add_scalar('Loss/train', avg_loss, epoch)
        writer_logs.add_scalar('Max_div/train', avg_accuracy, epoch)
        # evaluate model with validation set every 25 epochs and save checkpoint
        if epoch % 25 == 0:
            evaluate_model(model, val_dataloader, writer_logs, epoch, loss_fn, metric_fn, device, mode='val')
            checkpoint_save(model, savedir, epoch)
            # with tune.checkpoint_dir(epoch) as checkpoint_dir:
            #     path = os.path.join(checkpoint_dir, "checkpoint")
            #     os.makedirs(checkpoint_dir, exist_ok=True)
            #     torch.save((model.state_dict(), optimizer.state_dict()), path)

        # tune.report(loss=val_loss, accuracy=val_metric)

    # evaluate model with test set
    val_loss, val_metric = evaluate_model(model, test_dataloader, writer_logs, epoch, loss_fn, metric_fn, device, mode='test')

    # close tensorboard and save final model
    writer_logs.close()
    checkpoint_save(model, savedir, epoch)

def main():
    # load train config from yaml
    train_config = load_yaml("config/train_config.yaml")

    config = {
        "lr": 1e-3,
        "batch_size": 64,
    }

    train_cfderror(config, train_config, data_dir=train_config["data_dir"], checkpoint_dir=None)
    # # setup hyperparameter search space
    # config = {
    #     "lr": tune.loguniform(1e-5, 1e-1),
    #     "batch_size": tune.choice([8, 16, 32, 64, 128]),
    # }

    # # setup scheduler
    # scheduler = ASHAScheduler(
    #     metric="loss",
    #     mode="min",
    #     max_t=100,
    #     grace_period=1,
    #     reduction_factor=2)

    # # setup reporter
    # reporter = CLIReporter(
    #     parameter_columns=["lr", "batch_size"],
    #     metric_columns=["loss", "accuracy", "training_iteration"])

    # # run hyperparameter search
    # analysis = tune.run(
    #     partial(train_cfderror, train_config=train_config, data_dir=train_config["data_dir"]),
    #     resources_per_trial={"cpu": 16, "gpu": 1},
    #     config=config,
    #     num_samples=10,
    #     scheduler=scheduler,
    #     progress_reporter=reporter)
    
    # best_trial = analysis.get_best_trial("loss", "min", "last")
    # print("Best trial config: {}".format(best_trial.config))
    # print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    # print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

    # best_model = initialize_model(in_channel=train_config["in_channel"], out_channel=train_config["out_channel"], type=train_config["model_name"], layers=None, num_filters=None)
    # best_checkpoint_dir = best_trial.checkpoint.value
    # model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    # best_model.load_state_dict(model_state)

    # torch.save(best_model.state_dict(), os.path.join(best_checkpoint_dir, "best_model.pt"))

    # print("Best hyperparameters found were: ", analysis.best_config)
    # # save best hyperparameters to yaml
    # save_yaml(analysis.best_config, "config/best_config.yaml")

if __name__ == '__main__':
    main()