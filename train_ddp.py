# Filename: train.py
# Description: Train a model on multiple gpus for image classification task
# Author: Udari Madhushani

import os
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
import torchinfo
import torch.distributed as dist

from assets.models import CNNSimple
from assets.utils import get_dataloaders, get_metadata
from assets.trainers import VanillaTrainer

# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_ddp.py \
#   --configs ./assets/configs.yml --override_args dataset.name=cifar10 optimization.batch_size=512


def main():
    parser = argparse.ArgumentParser(description="Robust residual learning")

    parser.add_argument(
        "--configs",
        type=str,
        default="./assets/configs.yml",
        help="path to config file that provides common configs",
    )
    parser.add_argument(
        "--override_args",
        type=str,
        nargs="*",
        help=
        "orverride (or add new ones) specific configs values with cmd line args (dot-list format)",
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

    # Initialize ddp (LOCAL_RANK and WORLD_SIZE are set by torchrun)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["LOCAL_RANK"]),
    )
    # To maintain a common interface between single and multi-gpu scripts, we
    # assume that batch-size refers to total batch-size on all gpus. With multiple
    # gpus, we need to divide it by adjust batchsize and workers per gpu
    cfg.optimization.batch_size //= int(os.environ["WORLD_SIZE"])
    cfg.dataset.workers //= int(os.environ["WORLD_SIZE"])
    print(
        f"Per-gpu batch-size {cfg.optimization.batch_size}, num-workers {cfg.dataset.workers}"
    )

    # set device to "cuda:0" if gpu is available else "cpu". When using multiple
    # gpus in DistributedDataParallel, the device should be set to "cuda:local_rank"
    # so "cuda:0", "cuda:1", etc. where local_rank is the rank of the process.
    local_rank, num_devices = int(os.environ["LOCAL_RANK"]), int(
        os.environ["WORLD_SIZE"])
    device = f"cuda:{local_rank}"
    print("Current device: ", device)

    # set seed for reproducibility. Note that we want to set a different seed on each device
    np.random.seed(cfg.core.seed + local_rank)
    torch.manual_seed(cfg.core.seed + local_rank)
    torch.cuda.manual_seed(cfg.core.seed + local_rank)

    # set up training and validation dataloaders (TODO: Add sampler)
    metadata = get_metadata(cfg.dataset.name)
    sampler = lambda ds: torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=num_devices, rank=local_rank, shuffle=True)
    train_loader, val_loader = get_dataloaders(
        cfg.dataset.name,
        cfg.dataset.data_dir,
        metadata,
        cfg.optimization.batch_size,
        cfg.dataset.workers,
        sampler,
    )

    # set up model
    model = CNNSimple(
        in_channels=metadata.num_channels,  # type: ignore
        num_classes=metadata.num_classes,  # type: ignore
    ).to(device)  # type: ignore
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
    if local_rank == 0:
        print(
            torchinfo.summary(
                model,
                input_size=(
                    1,
                    metadata.num_channels,  # type: ignore
                    metadata.image_size,  # type: ignore
                    metadata.image_size,  # type: ignore
                ),  # type: ignore
                col_names=(
                    "input_size",
                    "output_size",
                    "num_params",
                    "kernel_size",
                    "mult_adds",
                ),
            ))

    # set up optimizers
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.optimization.lr,
        momentum=cfg.optimization.momentum,
        weight_decay=cfg.optimization.weight_decay,
    )
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
                             verbose=True if local_rank == 0 else False)
    trainer.train_all_epochs(cfg.optimization.epochs)


if __name__ == "__main__":
    main()
