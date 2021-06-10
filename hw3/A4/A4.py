import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()


np.random.seed(1968990 + 20210517)
torch.manual_seed(1968990 + 20210517)

# https://courses.cs.washington.edu/courses/cse446/21sp/sections/07/Pytorch_Neural_Networks.html

def preapare_dataset():

    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=256, shuffle=True)

    return train_loader, test_loader


# https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073
# stdv = 1. / math.sqrt(self.weight.size(1))
# self.weight.data.uniform_(-stdv, stdv)
# if self.bias is not None:
#     self.bias.data.uniform_(-stdv, stdv)
class MNISTNetwork_A4a(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_0 = nn.Linear(784, self.hidden_size)
        self.linear_1 = nn.Linear(self.hidden_size, 10)

    def forward(self, inputs):
        x = self.linear_0(inputs)
        x = F.relu(x)
        x = self.linear_1(x)
        return F.log_softmax(x, dim=1)


class MNISTNetwork_A4b(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_0 = nn.Linear(784, self.hidden_size)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_2 = nn.Linear(self.hidden_size, 10)

    def forward(self, inputs):
        x = self.linear_0(inputs)
        x = F.relu(x)
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)
        return F.log_softmax(x, dim=1)


def main_A4a():
    n = 28
    input_size = 784
    hidden_size = 64
    k = 10
    epochs = 32

    train_loader, test_loader = preapare_dataset()

    model = MNISTNetwork_A4a(hidden_size).to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=5e-3)

    print(sum(p.numel() for p in model.parameters()))

    model.train()
    losses = []
    for i in range(epochs):
        loss_epoch = 0
        acc = 0
        for batch in tqdm(train_loader):
            images, labels = batch
            images, labels = images.to("cuda"), labels.to("cuda")
            images = images.view(-1, 784)
            optimizer.zero_grad()
            logits = model(images)
            preds = torch.argmax(logits, 1)
            acc += torch.sum(preds == labels).item()
            loss = F.cross_entropy(logits, labels)
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch ", i)
        print("Loss:", loss_epoch / len(train_loader.dataset))
        print("Acc:", acc / len(train_loader.dataset))
        losses.append(loss_epoch / len(train_loader.dataset))
        if acc / len(train_loader.dataset) > 0.99:
            break

    loss_epoch = 0
    acc = 0
    for batch in tqdm(test_loader):
        images, labels = batch
        images, labels = images.to("cuda"), labels.to("cuda")
        images = images.view(-1, 784)

        logits = model(images)
        preds = torch.argmax(logits, 1)
        acc += torch.sum(preds == labels).item()
        loss = F.cross_entropy(logits, labels)
        loss_epoch += loss.item()
    print('Testing dataset')
    print("Loss:", loss_epoch / len(test_loader))
    print("Acc:", acc / len(test_loader.dataset))

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(losses)), losses, '--x', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('A4a.png')

def main_A4b():
    n = 28
    input_size = 784
    hidden_size = 32
    k = 10
    epochs = 64

    train_loader, test_loader = preapare_dataset()

    model = MNISTNetwork_A4b(hidden_size).to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=5e-3)

    print(sum(p.numel() for p in model.parameters()))

    model.train()
    losses = []
    for i in range(epochs):
        loss_epoch = 0
        acc = 0
        for batch in tqdm(train_loader):
            images, labels = batch
            images, labels = images.to("cuda"), labels.to("cuda")
            images = images.view(-1, 784)
            optimizer.zero_grad()
            logits = model(images)
            preds = torch.argmax(logits, 1)
            acc += torch.sum(preds == labels).item()
            loss = F.cross_entropy(logits, labels)
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch:", i)
        print("Loss:", loss_epoch / len(train_loader))
        print("Acc:", acc / len(train_loader.dataset))
        losses.append(loss_epoch / len(train_loader))
        if acc / len(train_loader.dataset) > 0.99:
            break

    loss_epoch = 0
    acc = 0
    for batch in tqdm(test_loader):
        images, labels = batch
        images, labels = images.to("cuda"), labels.to("cuda")
        images = images.view(-1, 784)

        logits = model(images)
        preds = torch.argmax(logits, 1)
        acc += torch.sum(preds == labels).item()
        loss = F.cross_entropy(logits, labels)
        loss_epoch += loss.item()
    print('Testing dataset')
    print("Loss:", loss_epoch / len(test_loader))
    print("Acc:", acc / len(test_loader.dataset))

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(losses)), losses, '--x', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('A4b.png')


if __name__ == '__main__':
    main_A4a()
    main_A4b()