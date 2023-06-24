import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNSimple(nn.Module):

    def __init__(self, in_channels=3, hidden_dim=16, num_classes=10):
        super().__init__()
        assert num_classes <= 10, "not recommended to use more than 10 classes with this model"
        self.conv1 = nn.Conv2d(in_channels,
                               hidden_dim,
                               kernel_size=5,
                               padding=2)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(hidden_dim,
                               hidden_dim,
                               kernel_size=5,
                               padding=2)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(hidden_dim,
                               hidden_dim,
                               kernel_size=5,
                               padding=2)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        self.pool3 = nn.AdaptiveAvgPool2d(
            4)  # no matter then input size, output will be 4x4 image
        self.fc1 = nn.Linear(hidden_dim * 4 * 4, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)  # [N, C, H, W] -> [N, C*H*W]
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # [N, num_classes]
        return x
