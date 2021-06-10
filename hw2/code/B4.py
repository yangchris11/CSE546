import torch
import torchvision

from tqdm import tqdm


def load_dataset():
    MNIST_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: torch.flatten(x))
        ]
    )
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=MNIST_transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=MNIST_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    return train_loader, test_loader


def eval(dataloader, W):
    correct = 0
    total = len(dataloader) * 256
    for X, Y in dataloader:
        Y_pred = torch.matmul(X, W)
        correct += torch.sum(torch.argmax(Y_pred, 1) == Y)
    return (correct/total).item()


def train_CE(epochs, step_size, train_loader, test_loader):
    W = torch.zeros(784, 10, requires_grad=True)
    for epoch in range(epochs):
        correct = 0
        total = len(train_loader) * 256
        for X_train, Y_train in tqdm(train_loader):
            Y_pred = torch.matmul(X_train, W)
            # cross entropy combines softmax calculation with NLLLoss
            loss = torch.nn.functional.cross_entropy(Y_pred, Y_train)
            # computes derivatives of the loss with respect to W
            loss.backward()
            # gradient descent update
            W.data = W.data - step_size * W.grad
            # .backward() accumulates gradients into W.grad instead
            # of overwriting, so we need to zero out the weights
            W.grad.zero_()
            correct += torch.sum(torch.argmax(Y_pred, 1) == Y_train)
            # print(loss)
        print('Training Acc: {} | Testing Acc: {}'.format((correct/total).item(), eval(test_loader, W)))


def train_MSE(epochs, step_size, train_loader, test_loader):
    W = torch.zeros(784, 10, requires_grad=True)
    for epoch in range(epochs):
        correct = 0
        total = len(train_loader) * 256
        for X_train, Y_train in tqdm(train_loader):
            Y_pred = torch.matmul(X_train, W)
            loss = torch.nn.functional.mse_loss(Y_pred, torch.nn.functional.one_hot(Y_train, 10).type(torch.FloatTensor))
            loss.backward()
            W.data = W.data - step_size * W.grad
            W.grad.zero_()
            correct += torch.sum(torch.argmax(Y_pred, 1) == Y_train)
            # print(loss)
        print('Training Acc: {} | Testing Acc: {}'.format((correct / total).item(), eval(test_loader, W)))


def main():
    train_loader, test_loader = load_dataset()
    # train_CE(30, 0.05, train_loader, test_loader)
    train_MSE(30, 0.05, train_loader, test_loader)


if __name__ == '__main__':
    main()