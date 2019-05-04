
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        #self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(512, 512)
        self.drop1 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        X = x
        X = self.resnet(X)
        X = F.relu(self.drop1(X))
        X = F.relu(self.drop2(self.fc1(X)))
        # X = torch.cat([X, meta], dim=1)
        X = self.fc2(X)

        return F.log_softmax(X, dim=1)