
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class CustomResnetV1(nn.Module):

    def __init__(self):
        super(CustomResnetV1, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False)
        self.resnet.fc = nn.Linear(512, 512)

        self.fc11 = nn.Linear(512, 512)
        self.fc12 = nn.Linear(512, 512)
        self.fc_down1 = nn.Linear(512, 256)

        self.fc21 = nn.Linear(256, 256)
        self.fc22 = nn.Linear(256, 256)
        self.fc_down2 = nn.Linear(256, 128)

        self.fc31 = nn.Linear(128, 128)
        self.fc32 = nn.Linear(128, 128)

        self.fc4 = nn.Linear(128, 10)

        #self.drop1 = nn.Dropout2d(0.5)

    def forward(self, x):
        x_ = F.relu(self.resnet(x))

        x = F.relu(self.fc11(x_))
        x = F.relu(self.fc12(x))
        x_ = self.fc_down1(torch.add(x, x_))

        x = F.relu(self.fc21(x_))
        x = F.relu(self.fc22(x))
        x_ = self.fc_down2(torch.add(x, x_))

        x = F.relu(self.fc31(x_))
        x = F.relu(self.fc32(x))
        x_ = torch.add(x, x_)

        x = self.fc4(x_)

        return F.log_softmax(x, dim=1)
