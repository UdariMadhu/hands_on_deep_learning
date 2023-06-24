# Filename: train.py
# Description: Train a model on a single gpu for image classification task
# Author: Udari Madhushani

import os
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
import torchinfo

from assets.models import CNNSimple
from assets.utils import get_dataloaders, get_metadata
from assets.trainers import VanillaTrainer

# Usage: CUDA_VISIBLE_DEVICES=0 python train.py --configs ./assets/configs.yml --override_args dataset.name=cifar10

def main():
    parser = argparse.ArgumentParser(description="Robust residual learning")

    parser.add_argument(
        "--configs",
        type=str,
        default="./assets/configs.yml",
        help="path to config file that provides common configs")
    parser.add_argument(
        '--override_args',
        type=str,
        nargs="*",
        help=
        'orverride (or add new ones) specific configs values with cmd line args (dot-list format)'
    )

    args = parser.parse_args()  # parse command line args first
    cfg = OmegaConf.load(args.configs)  # load args from config file
    # override args from config file with command line args
    if args.override_args is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override_args))
    cfg = OmegaConf.merge(cfg, OmegaConf.create(vars(args)))  # add args to cfg
    print(cfg)

    # setup logging
    os.makedirs(cfg.core.save_dir, exist_ok=True)

    # set device to "cuda:0" if gpu is available else "cpu". When using multiple
    # gpus in DistributedDataParallel, the device should be set to "cuda:local_rank"
    # so "cuda:0", "cuda:1", etc. where local_rank is the rank of the process.
    device = "cuda:0"

    # set seed for reproducibility
    np.random.seed(cfg.core.seed)
    torch.manual_seed(cfg.core.seed)
    torch.cuda.manual_seed(cfg.core.seed)

    # set up training and validation dataloaders
    metadata = get_metadata(cfg.dataset.name)
    train_loader, val_loader = get_dataloaders(cfg.dataset.name,
                                               cfg.dataset.data_dir, metadata,
                                               cfg.optimization.batch_size,
                                               cfg.dataset.workers)

    # set up model
    model = CNNSimple(
        in_channels=metadata.num_channels,  # type: ignore
        num_classes=metadata.num_classes)  # type: ignore
    model = model.to(device)  # move model to device
    print(
        torchinfo.summary(
            model,
            input_size=(
                1,
                metadata.num_channels,  # type: ignore
                metadata.image_size,  # type: ignore
                metadata.image_size),  # type: ignore
            col_names=("input_size", "output_size", "num_params",
                       "kernel_size", "mult_adds"),
        ))

    # set up optimizers
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.optimization.lr,
                                momentum=cfg.optimization.momentum,
                                weight_decay=cfg.optimization.weight_decay)
    # set up learning rate scheduler (decay learning rate after each epochs)
    # One can also decay learning rate after each batch, where the total number of steps
    # would be num_epochs * num_batches_per_epoch in the cosine annealing scheduler. There
    # one would call lrs.step() after each batch. We call it only after each epoch for simplicity.
    lrs = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.optimization.epochs, eta_min=0.0)
    loss_fxn = torch.nn.CrossEntropyLoss()

    # Let's train the model!!!
    trainer = VanillaTrainer(model,
                             train_loader,
                             val_loader,
                             optimizer,
                             loss_fxn,
                             lrs,
                             device,
                             cfg.core.print_freq,
                             verbose=True)
    trainer.train_all_epochs(cfg.optimization.epochs)


if __name__ == "__main__":
    main()
