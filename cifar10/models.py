
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import modules
import custom


def build_custom_v1():
    return nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(512, 1024, kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(2),
            nn.ReLU(),

            modules.Flatten(),

            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(512, 10),
            nn.LogSoftmax(dim=1)
    )


def build_cifar10_resnet(resnet, fc_size):
    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False)
    resnet.fc = nn.Linear(fc_size, 512)
    return nn.Sequential(
            resnet,
            nn.Dropout2d(),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.LogSoftmax(dim=1)
    )


def get_model(name='resnet18', pretrained=True):
    name = name.lower()
    if name == 'resnet18':
        return build_cifar10_resnet(torchvision.models.resnet18(pretrained=pretrained), fc_size=512)

    elif name == 'resnet34':
        return build_cifar10_resnet(torchvision.models.resnet34(pretrained=pretrained), fc_size=512)

    elif name == 'resnet50':
        return build_cifar10_resnet(torchvision.models.resnet50(pretrained=pretrained), fc_size=2048)

    elif name == 'custom-1':
        return build_custom_v1()

    elif name == 'custom-2':
        return custom.CustomResnetV1()

