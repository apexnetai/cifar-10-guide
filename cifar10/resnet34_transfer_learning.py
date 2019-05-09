
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

        self.bn1a = nn.BatchNorm1d(256)
        self.fc11 = nn.Linear(256, 256)
        self.fc12 = nn.Linear(256, 256)
        self.bn1b = nn.BatchNorm1d(256)
        self.fc13 = nn.Linear(256, 256)
        self.fc14 = nn.Linear(256, 256)
        self.bn1c = nn.BatchNorm1d(256)
        self.fc15 = nn.Linear(256, 256)
        self.fc16 = nn.Linear(256, 256)
        self.fc_down1 = nn.Linear(256, 128)

        self.bn2a = nn.BatchNorm1d(128)
        self.fc21 = nn.Linear(128, 128)
        self.fc22 = nn.Linear(128, 128)
        self.bn2b = nn.BatchNorm1d(128)
        self.fc23 = nn.Linear(128, 128)
        self.fc24 = nn.Linear(128, 128)
        self.bn2c = nn.BatchNorm1d(128)
        self.fc25 = nn.Linear(128, 128)
        self.fc26 = nn.Linear(128, 128)
        self.fc_down2 = nn.Linear(128, 64)

        self.bn3a = nn.BatchNorm1d(64)
        self.fc31 = nn.Linear(64, 64)
        self.fc32 = nn.Linear(64, 64)
        self.bn3b = nn.BatchNorm1d(64)
        self.fc33 = nn.Linear(64, 64)
        self.fc34 = nn.Linear(64, 64)
        self.bn3c = nn.BatchNorm1d(64)
        self.fc35 = nn.Linear(64, 64)
        self.fc36 = nn.Linear(64, 64)

        self.fc4 = nn.Linear(64, 10)

        #self.drop1 = nn.Dropout2d(0.5)

    def forward(self, x):
        x_ = F.relu(self.resnet(x))

        x = self.bn1a(x_)
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        x_ = torch.add(x, x_)
        x = self.bn1b(x_)
        x = F.relu(self.fc13(x))
        x = F.relu(self.fc14(x))
        x_ = torch.add(x, x_)
        x = self.bn1c(x_)
        x = F.relu(self.fc15(x))
        x = F.relu(self.fc16(x))
        x_ = self.fc_down1(torch.add(x, x_))

        x = self.bn2a(x_)
        x = F.relu(self.fc21(x))
        x = F.relu(self.fc22(x))
        x_ = torch.add(x, x_)
        x = self.bn2b(x_)
        x = F.relu(self.fc23(x))
        x = F.relu(self.fc24(x))
        x_ = torch.add(x, x_)
        x = self.bn2c(x_)
        x = F.relu(self.fc25(x))
        x = F.relu(self.fc26(x))
        x_ = self.fc_down2(torch.add(x, x_))

        x = self.bn3a(x_)
        x = F.relu(self.fc31(x))
        x = F.relu(self.fc32(x))
        x_ = torch.add(x, x_)
        x = self.bn3b(x_)
        x = F.relu(self.fc33(x))
        x = F.relu(self.fc34(x))
        x_ = torch.add(x, x_)
        x = self.bn3c(x_)
        x = F.relu(self.fc35(x))
        x = F.relu(self.fc36(x))
        x_ = torch.add(x, x_)

        x = self.fc4(x_)

        return F.log_softmax(x, dim=1)
