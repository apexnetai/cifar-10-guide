
import timeit
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse

from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from tqdm import tqdm

import models


def train(model, device, train_loader, optimizer, epoch, train_losses=None, log_interval=1):
    if train_losses is None:
        train_losses = []

    model.train()
    data_with_progress = tqdm(enumerate(train_loader), total=len(train_loader)) 
    for batch_idx, (data, target) in data_with_progress: 
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            desc = f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss({loss.item():.6f})'
            data_with_progress.set_description(desc=desc)


def test(model, device, test_loader, test_losses=[]):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += loss
            test_losses.append(loss / data.shape[0])
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))



def learn(model, optimizer, train_loader, test_loader,
          epochs=10, train_losses=None, test_losses=None, log_interval=10, train_sample_loader=None):
    if train_losses is None:
        train_losses = []

    if test_losses is None:
        test_losses = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, train_losses, log_interval)
        test(model, device, test_loader, test_losses)
        if train_sample_loader is not None:
            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in train_sample_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

            print('Train set accuracy: Accuracy: {}/{} ({:.0f}%)\n'.format(
                    correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))



def cifar10(training=True, batch_size=64, transforms=None):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    return torch.utils.data.DataLoader(
            datasets.CIFAR10('./data',
                             train=training,
                             download=True,
                             transform=transforms),
            batch_size=batch_size,
            shuffle=True,
            **kwargs)



def main():
    train_losses = []
    test_losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_model = models.get_model(name='custom-2') #model.Model().to(device)
    optimizer = optim.SGD(learning_model.parameters(), lr=0.1, momentum=0.5)


    seed = 42

    torch.manual_seed(seed)

    batch_size = 128
    test_batch_size = 128
    tensorNormalz = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
    ])

    train_loader = cifar10(training=True, batch_size=batch_size, transforms=tensorNormalz)
    test_loader = cifar10(training=False, batch_size=test_batch_size, transforms=tensorNormalz)

    np.random.seed(seed)
    indices = np.random.randint(low=0, high=len(train_loader.dataset), size=len(test_loader.dataset))
    train_sample_loader = torch.utils.data.DataLoader(Subset(train_loader.dataset, indices),
                                                      test_loader.batch_size,
                                                      drop_last=False)
    params = {
        'epochs': 16,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'log_interval': 8,
        'train_sample_loader': train_sample_loader
    }

    learn(learning_model, optimizer, train_loader, test_loader, **params)



if __name__ == '__main__':
    total_secs = timeit.timeit(main, number=1)
    millis = total_secs - int(total_secs)
    total_secs = int(total_secs)
    mins = total_secs // 60
    secs = total_secs % 60
    print(f'Total time: {mins}m {secs+millis:.2f}s\n')

