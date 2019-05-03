
import torch.nn as nn
import torch.nn.functional as F


class Standard(nn.Module):
    def __init__(self):
        super(Standard, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3) # change the input channels to 3 since cifar is colored
        self.conv_bn_1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3)
        self.conv_bn_2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3)
        self.conv_bn_3 = nn.BatchNorm2d(1024)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.conv3_drop = nn.Dropout2d(p=0.8)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc1a = nn.Linear(1024, 1024)
        self.fc1b = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv_bn_1(self.conv1(x)), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv_bn_2(self.conv2(x))), 2))
        # x = F.relu(F.max_pool2d(self.conv3_drop(self.conv_bn_3(self.conv3(x))), 2))
        x = F.relu(F.max_pool2d(self.conv_bn_2(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv_bn_3(self.conv3(x)), 2))
        x = x.view(x.size(0), -1) # we have to stretch out the data while maintaining the batch size
        x = F.dropout(F.relu(self.fc1(x)), p=0.4, training=self.training)
        x = F.dropout(F.relu(self.fc1a(x)), p=0.5, training=self.training)
        x = F.dropout(F.relu(self.fc1b(x)), p=0.6, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)