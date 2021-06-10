import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import transforms
from tqdm import tqdm

import time

from mnist import MNIST

sns.set()
plt.rcParams["axes.grid"] = False

torch.manual_seed(1968990 + 20210604)
np.random.seed(1968990 + 20210604)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AutoEncoder(nn.Module):
    def __init__(self, input_size=784, hidden_size=64):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.input_size)
        )

    def forward(self, x):
        z = self.encoder(x)
        r = self.decoder(z)
        return r


class AutoEncoder_relu(nn.Module):
    def __init__(self, input_size=784, hidden_size=64):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.input_size),
            nn.ReLU()
        )

    def forward(self, x):
        z = self.encoder(x)
        r = self.decoder(z)
        return r


def load_dataset():
    mnist_trainset = datasets.MNIST(root='./data/', train=True, download=True, transform=transforms.ToTensor())
    mnist_testset = datasets.MNIST(root='./data/', train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=16, shuffle=False)

    mnist_visset = datasets.MNIST(root='./data/', train=False, download=True, transform=transforms.ToTensor())
    current_digit = 0
    mnist_visset.data = []
    mnist_visset.targets = []
    for data, label in zip(mnist_trainset.data, mnist_trainset.targets):
        if label == current_digit:
            mnist_visset.data.append(data)
            mnist_visset.targets.append(label)
            current_digit += 1
        if label >= 10:
            break

    vis_loader = torch.utils.data.DataLoader(mnist_visset, batch_size=10, shuffle=False)

    return train_loader, test_loader, vis_loader


def train(epochs, data_loader, model, optimizer, criterion):
    for epoch in range(epochs):
        loss = 0
        for i, (data, _) in enumerate(data_loader):
            data = data.view(-1, 28*28).to(DEVICE)
            optimizer.zero_grad()

            outputs = model(data)

            train_loss = criterion(outputs, data)
            train_loss.backward()

            optimizer.step()

            loss += train_loss.item()

        loss /= len(data_loader)
        if epoch + 1 == epochs:
            print("Epoch : {}/{}, Train loss = {:.6f}".format(epoch + 1, epochs, loss))


def test(data_loader, model, criterion=nn.MSELoss()):
    loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):

            data = data.view(-1, 28*28).to(DEVICE)
            outputs = model(data)
            test_loss = criterion(outputs, data)
            loss += test_loss.item()

        loss /= len(data_loader)
        print("Test loss = {:.6f}".format(loss))

def plot_original_and_reconstruction(data_loader, model, save_path):

    for data, labels in data_loader:
        data = data.view(-1, 28*28).to(DEVICE)
        with torch.no_grad():
            original = data
            reconstruct = model(data)

    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(10, 3))
    for i, row in enumerate(axes):
        for j, cell in enumerate(row):
            if i == 0:
                cell.imshow(original.cpu().data[j, :].reshape((28, 28)))
                cell.set_title(j)
                if j == 0:
                    cell.set_ylabel('Original')
                cell.set(xticklabels=[])
                cell.set(yticklabels=[])
            else:
                cell.imshow(reconstruct.cpu().data[j, :].reshape((28, 28)))
                if j == 0:
                    cell.set_ylabel('Reconstructed')
    plt.tight_layout()
    plt.savefig(save_path)


def main_A4a(ks=[32, 64, 128]):

    train_loader, test_loader, vis_loader = load_dataset()

    for k in ks:
        model = AutoEncoder(hidden_size=k).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        train(15, train_loader, model, optimizer, criterion)

        plot_original_and_reconstruction(vis_loader, model, 'A4a_{}.png'.format(k))

        test(test_loader, model, criterion)

    return model


def main_A4b(ks=[32, 64, 128]):

    train_loader, test_loader, vis_loader = load_dataset()

    for k in ks:
        model = AutoEncoder_relu(hidden_size=k).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        train(15, train_loader, model, optimizer, criterion)

        plot_original_and_reconstruction(vis_loader, model, 'A4b_{}.png'.format(k))

        test(test_loader, model, criterion)

    return model


if __name__ == '__main__':
    model_a = main_A4a()
    model_b = main_A4b()
