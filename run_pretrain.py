# Should be imported primarily
from tools import random_seed

import argparse
from argparse import Namespace
import os
import yaml
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision
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
    validation_writer = SummaryWriter(os.path.join(output_path, "validation"))


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
    decay_factor = float(config['solver']['weight_decay']['scale'])
    weight_decay = float(config['solver']['weight_decay']['weight'])
    bias_decay = float(config['solver']['weight_decay']['bias'])
    milestones = config['solver']['milestones']

    print("\n[ Model & Solver ]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_params.device = device
    print(f'--- Device: {device}')
    model = models.__dict__[model_type]().to(device)
    print(f"--- Model architecture: {model_type}")
    
    assert solver_type in ['adam', 'sgd']
    print(f"--- Solver type: {solver_type}")
    param_groups = [
        {"params": model.bias_parameters(), "weight_decay": bias_decay},
        {"params": model.weight_parameters(), "weight_decay": weight_decay}
    ]

    if device.type == "cuda":
        model = torch.nn.DataParallel(model).cuda()
        # cudnn.benchmark = True

    if solver_type == "adam":
        optimizer = torch.optim.Adam(param_groups, learning_rate, betas=(alpha, beta))
    elif solver_type == "sgd":
        optimizer = torch.optim.SGD(param_groups, learning_rate, momentum=alpha)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=decay_factor)


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
            validation_EPE = validate(config, training_params, data_loader.validation_set_loader, model, epoch, validation_writer)
        validation_writer.add_scalar("mean EPE", validation_EPE, epoch)

        if training_params.best_EPE < 0:
            training_params.best_EPE = validation_EPE

        is_best = validation_EPE < training_params.best_EPE
        best_EPE = min(validation_EPE, training_params.best_EPE)
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
    train_writer.close()


def train(config, params, data_loader, model, optimizer, epoch, train_writer):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    flow2_EPEs = utils.AverageMeter()

    epoch_size = len(data_loader) if config['epoch_size'] == 0 else min(len(data_loader), config['epoch_size'])
    model.train()
    end = time.time()

    for batch_idx, (input, target) in enumerate(data_loader):
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

        # compute gradient and do optimize
        optimizer.zero_grad()
        loss.backward()

        batch_time.update(time.time() - end)
        end = time.time()

        # ========== Log training infos ========== 
        if batch_idx % config['training_log_rate'] == 0:
            print("Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}\t EPE {6}".format(
                    epoch, batch_idx, epoch_size, batch_time, data_time, losses, flow2_EPEs))
        if config['tensorboard']['learning_rate']['enable'] is True:
            if params.num_iter % config['tensorboard']['learning_rate']['log_rate'] == 0:
                bias_learning_rate = optimizer.param_groups[0]['lr']
                weight_learning_rate = optimizer.param_groups[1]['lr']
                train_writer.add_scalar("learning_rate/bias", bias_learning_rate, params.num_iter)
                train_writer.add_scalar("learning_rate/weight", weight_learning_rate, params.num_iter)
        if config['tensorboard']['weight_and_bias']['enable'] is True:
            if params.num_iter % config['tensorboard']['weight_and_bias']['log_rate'] == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        train_writer.add_histogram(f"weight_and_bias/{name}", param, params.num_iter)
        if config['tensorboard']['gradient']['enable'] is True:
            if params.num_iter % config['tensorboard']['gradient']['log_rate'] == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        train_writer.add_histogram(f"gradients/{name}", param.grad, params.num_iter)

        optimizer.step()
        params.num_iter += 1
        if batch_idx >= epoch_size:
            break
    return losses.avg, flow2_EPEs.avg

    

def validate(config, params, data_loader, model, epoch, validation_writer):
    batch_time = utils.AverageMeter()
    flow2_EPEs = utils.AverageMeter()

    model.eval()

    end = time.time()
    for batch_idx, (input, target) in enumerate(data_loader):
        target = target.to(params.device)
        input = torch.cat(input, 1).to(params.device)

        # compute output
        output = model(input)
        flow2_EPE = config['div_flow'] * loss_functions.realEPE(output, target, sparse=False)
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if config['tensorboard']['sample_output']['enable'] is True:
            if batch_idx < config['tensorboard']['sample_output']['num_samples']:
                mean_values = torch.tensor([0.45, 0.432, 0.411], dtype=input.dtype).view(3, 1, 1)
                validation_writer.add_image(
                    "GroundTruth/"+str(batch_idx), utils.flow2rgb(config['div_flow'] * target[0], max_value=10), 0)
                validation_writer.add_image(
                    "Inputs/"+str(batch_idx), (input[0, :3].cpu() + mean_values).clamp(0, 1), 0)
                validation_writer.add_image(
                    "Inputs/"+str(batch_idx), (input[0, 3:].cpu() + mean_values).clamp(0, 1), 1)
                validation_writer.add_image(
                    "FlowNet Result/"+str(batch_idx), utils.flow2rgb(config['div_flow'] * output[0], max_value=10), epoch)

        if batch_idx % config['training_log_rate'] == 0:
            print("Test: [{0}/{1}]\t Time {2}\t EPE {3}".format(
                    batch_idx, len(data_loader), batch_time, flow2_EPEs))
    print(" * EPE {:.3f}".format(flow2_EPEs.avg))
    return flow2_EPEs.avg


if __name__ == "__main__":
    main()
