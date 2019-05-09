
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class CustomResnetV1(nn.Module):

    def __init__(self):
        super(CustomResnetV1, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False)
        self.resnet.fc = nn.Linear(512, 256)

        self.block1 = self.make_block(256)
        self.fc_down1 = nn.Linear(256, 128)

        self.block2 = self.make_block(128)
        self.fc_down2 = nn.Linear(128, 64)

        self.block3 = self.make_block(64)

        self.fc4 = nn.Linear(64, 10)


    def forward(self, x):
        x_ = F.relu(self.resnet(x))
        x_ = self._forward_block(x_, self.block1, self.fc_down1)
        x_ = self._forward_block(x_, self.block2, self.fc_down2)
        x_ = self._forward_block(x_, self.block3)
        x = self.fc4(x_)

        return F.log_softmax(x, dim=1)


    def make_block(self, size):
        blocks = [self.make_sub_block(size) for _ in range(3)]
        return nn.Sequential(*blocks)


    @staticmethod
    def make_sub_block(size):
        return nn.Sequential(
                nn.BatchNorm1d(size),
                nn.Linear(size, size),
                nn.ReLU(),
                nn.Linear(size, size),
                nn.ReLU()
        )


    @staticmethod
    def _forward_block(x_, blocks, downsample=None):
        for block in blocks:
            x = block(x_)
            x_ = torch.add(x, x_)
        if downsample is not None:
            x_ = downsample(x_)
        return x_
