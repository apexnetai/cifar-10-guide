
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False)
        self.resnet.fc = nn.Linear(512, 512)
        self.drop1 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(self.drop1(x))
        x = F.relu(self.drop2(self.fc1(x)))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
