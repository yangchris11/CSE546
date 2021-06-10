import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

sns.set()

np.random.seed(1968990 + 20210518)
torch.manual_seed(1968990 + 20210518)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

batch_size = 256


def prepare_dataset():
    cifar10_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_size = int(len(cifar10_set) * 0.9)
    val_size = len(cifar10_set) - train_size
    cifar10_trainset, cifar10_valset = torch.utils.data.random_split(cifar10_set, [train_size, val_size])
    cifar10_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(cifar10_trainset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(cifar10_valset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(cifar10_testset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

def eval(model, dataloader, epoch):
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        running_loss = 0.0
        for data in dataloader:
            images, labels = data[0].to("cuda"), data[1].to("cuda")
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

    print('[%d] Val Accuracy: %.5f %% Val Loss: %.5f' % (epoch + 1, 100 * correct / total, running_loss / total))
    return running_loss / total

def run(model, exp='A5a'):
    train_loader, val_loader, test_loader = prepare_dataset()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-5)
    epochs = 16

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        running_loss = 0.0
        acc = 0
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            inputs, labels = data[0].to("cuda"), data[1].to("cuda")
            optimizer.zero_grad()

            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('[%d] Train Loss: %.5f' % (epoch + 1, running_loss / (len(train_loader) * batch_size)))
        train_losses.append(running_loss / (len(train_loader) * batch_size))
        running_loss = 0.0
        val_losses.append(eval(model, val_loader, epoch))

    eval(model, test_loader, -2)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, epochs + 1), train_losses, '--x', label='Training Loss')
    plt.plot(np.arange(1, epochs + 1), val_losses, '--o', label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(exp + '.png')


def main():
    model = torchvision.models.alexnet(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, 10)
    model.to('cuda')
    run(model, 'A5a')
    model = torchvision.models.alexnet(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 10)
    model.to('cuda')
    run(model, 'A5b')


if __name__ == '__main__':
    main()