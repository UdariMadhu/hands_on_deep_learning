import os
import glob
import torch
import numpy as np
from PIL import Image
from easydict import EasyDict
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def get_metadata(name):
    if name == "mnist":
        metadata = EasyDict({
            "image_size": 28,
            "num_classes": 10,
            "train_images": 60000,
            "val_images": 10000,
            "num_channels": 1,
        })
    elif name == "cifar10":
        metadata = EasyDict({
            "image_size": 32,
            "num_classes": 10,
            "train_images": 50000,
            "val_images": 10000,
            "num_channels": 3,
        })
    elif name == "imagenet100":
        metadata = EasyDict({
            "image_size": 224,
            "num_classes": 100,
            "train_images": 128000,
            "val_images": 5000,
            "num_channels": 3,
        })
    else:
        raise ValueError(f"{name} dataset not supported!")
    return metadata


def get_dataset(name, data_dir, metadata):
    """
    Return a dataset with the current name. We only support two datasets with
    their fixed image resolutions. One can easily add additional datasets here.
    """

    if name == "mnist":
        transform_train = transforms.Compose([
            transforms.RandomCrop(metadata.image_size, padding=4),
            transforms.ToTensor()
        ])
        transform_val = transforms.ToTensor()
        train_set = datasets.MNIST(root=data_dir,
                                   train=True,
                                   download=True,
                                   transform=transform_train)
        val_set = datasets.MNIST(root=data_dir,
                                 train=False,
                                 download=True,
                                 transform=transform_val)
    elif name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(metadata.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_val = transforms.ToTensor()
        train_set = datasets.CIFAR10(root=data_dir,
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        val_set = datasets.CIFAR10(root=data_dir,
                                   train=False,
                                   download=True,
                                   transform=transform_val)

    elif name in ["imagenet100"]:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(metadata.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_val = transforms.Compose([
            transforms.Resize(int(1.2 * metadata.image_size)),
            transforms.CenterCrop(metadata.image_size),
            transforms.ToTensor(),
        ])
        train_set = ImageNet100Dataset(os.path.join(data_dir, "train"),
                                       transform=transform_train)
        val_set = ImageNet100Dataset(os.path.join(data_dir, "train"),
                                     transform=transform_val)
    else:
        raise ValueError(f"{name} dataset nor supported!")
    return train_set, val_set


def get_dataloaders(name, data_dir, metadata, batch_size, num_workers=4, sampler=None):
    """
    Create train and validation dataloaders (provide sampler when using distributed ddp training).
    """
    train_set, val_set = get_dataset(name, data_dir, metadata)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True if sampler is None else False,
                                               num_workers=num_workers,
                                               pin_memory=True, 
                                               sampler=sampler(train_set))
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             pin_memory=True, 
                                             sampler=sampler(val_set))
    return train_loader, val_loader


class ImageNet100Dataset(Dataset):
    """
    A custom PyTorch Dataset for ImageNet100 that considers every 10th class from the imagenet (1k classes).

    Args:
        root_dir (str): Path to the root directory containing the ImageNet dataset.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
                                       Default is None.

    Attributes:
        root_dir (str): Path to the root directory containing the ImageNet dataset.
        transform (callable): A function/transform that takes in a PIL image and returns a transformed version.
        image_paths (list): List of paths to all the images in the dataset.
        labels (list): List of corresponding labels for the images in the dataset.

    Code generated with the following prompt from gpt-4: write a code for a custom pytorch dataset for imagenet100. 
    It uses dir with all imagenet images but only considers every 10th class
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.labels = self._load_data()

    def _load_data(self):
        image_paths = []
        labels = []

        class_dirs = sorted(glob.glob(os.path.join(self.root_dir, '*')))
        selected_class_dirs = class_dirs[::
                                         10]  # Select every 10th class directory

        for label, class_dir in enumerate(selected_class_dirs):
            # Assuming all imagenet images have .JPEG extension
            class_image_paths = glob.glob(os.path.join(class_dir, '*.JPEG'))
            image_paths.extend(class_image_paths)
            labels.extend([label] * len(class_image_paths))

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label
