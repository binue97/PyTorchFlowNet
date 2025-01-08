# Should be imported primarily
from tools import random_seed

import argparse
from argparse import Namespace
import os
import yaml
import time
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import models
from tools import flow_transforms
from tools import utils
from custom_datasets import FlyingChairsDataLoader
from tools import loss_functions


training_params = Namespace(
    device="",
    num_iter=0,
    best_EPE=-1
)

def main():
    print('')
    print("\n[ Configure ]")
    parser = argparse.ArgumentParser(description="Read a YAML config file.")
    parser.add_argument("-i", "--config", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()


    # ========== Read YAML file ==========
    print(f"--- Config file path: {args.config}")
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    print("--- Done reading config file")


    # ========== Results ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config["result_path"] , timestamp)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print(f"--- Output path: {output_path}")
    config["result_path"] = output_path
    
    train_writer = SummaryWriter(os.path.join(output_path, "train"))
    test_writer = SummaryWriter(os.path.join(output_path, "test"))


    # ========== Data Loader ==========
    print("\n[ Data Loader ]")
    dataset_type = config['dataset']['type']
    if dataset_type == 'flying_chairs':
        data_loader = FlyingChairsDataLoader(config)
  

    # ========== Create Model and Solver ==========
    solver_type = config['solver']['type']
    model_type = config['model_arch']
    alpha = float(config['solver']['alpha'])  # or momentum
    beta = float(config['solver']['beta'])
    learning_rate = float(config['solver']['learning_rate'])
    learning_rate_decay = float(config['solver']['learning_rate_decay'])
    milestones = config['solver']['milestones']

    print("\n[ Model & Solver ]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_params.device = device
    print(f'--- Device: {device}')
    model = models.__dict__[model_type]().to(device)
    print(f"--- Model architecture: {model_type}")
    
    assert solver_type in ['adam', 'sgd']
    print(f"--- Solver type: {solver_type}")
    params_groups = [
        {"params": model.bias_parameters(), "bias_decay": float(config['solver']['bias_decay'])},
        {"params": model.weight_parameters(), "weight_decay": float(config['solver']['weight_decay'])}
    ]

    if device.type == "cuda":
        model = torch.nn.DataParallel(model).cuda()
        # cudnn.benchmark = True

    if solver_type == "adam":
        optimizer = torch.optim.Adam(params_groups, learning_rate, betas=(alpha, beta))
    elif solver_type == "sgd":
        optimizer = torch.optim.SGD(params_groups, learning_rate, momentum=alpha)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=learning_rate_decay)


    # ========== Train and Validate ==========
    print("\n[ Train & Validate ]")
    training_params.best_EPE = -1
    for epoch in range(config['start_epoch'], config['epochs']):
        # Train
        train_loss, train_EPE = train(config, training_params, data_loader.training_set_loader, model, optimizer, epoch, train_writer)
        scheduler.step()
        train_writer.add_scalar("mean EPE", train_EPE, epoch)

        # Validate
        with torch.no_grad():
            EPE = validate(data_loader.test_set_loader, model, epoch)
        test_writer.add_scalar("mean EPE", EPE, epoch)

        if training_params.best_EPE < 0:
            training_params.best_EPE = EPE

        is_best = EPE < training_params.best_EPE
        best_EPE = min(EPE, training_params.best_EPE)
        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": model_type,
                "state_dict": model.module.state_dict(),
                "best_EPE": best_EPE,
                "div_flow": config['div_flow'],
            },
            is_best,
            output_path,
        )


def train(config, params, data_loader, model, optimizer, epoch, train_writer):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    flow2_EPEs = utils.AverageMeter()

    epoch_size = len(data_loader) if config['epoch_size'] == 0 else min(len(data_loader), config['epoch_size'])
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.to(params.device)

        input = torch.cat(input, 1).to(params.device)
        output = model(input)

        loss = loss_functions.multiscaleEPE(output, target, weights=config['solver']['multiscale_weights'], sparse=False)
        flow2_EPE = config['div_flow'] * loss_functions.realEPE(output[0], target, sparse=False)
        
        # record loss and EPE
        losses.update(loss.item(), target.size(0))
        train_writer.add_scalar("train_loss", loss.item(), params.num_iter)
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config['training_log_rate'] == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}\t EPE {6}".format(
                    epoch, i, epoch_size, batch_time, data_time, losses, flow2_EPEs
                )
            )
        params.num_iter += 1
        if i >= epoch_size:
            break
    return losses.avg, flow2_EPEs.avg

    

def validate(test_set_loader, model, epoch):
    batch_time = utils.AverageMeter()
    flow2_EPEs = utils.AverageMeter()
    return None

if __name__ == "__main__":
    main()
