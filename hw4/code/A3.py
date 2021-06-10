import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mnist import MNIST

sns.set()

np.random.seed(1968990 + 20210531)


def load_dataset():
    mnist_data = MNIST('./mnist/')
    # X, Y = map(np.array, mnist_data.load_training())
    X_train, Y_train = map(np.array, mnist_data.load_training())
    X_test, Y_test = map(np.array, mnist_data.load_testing())
    # X_train, Y_train = X[:50000], Y[:50000]
    # X_test, Y_test = X[50000:], Y[50000:]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, Y_train, X_test, Y_test


def main():
    X_train, Y_train, X_test, Y_test = load_dataset()
    n, d = X_train.shape
    m, d = X_test.shape
    # print(X_train.shape)
    # print(X_test.shape)
    I = np.ones((n, 1))

    mu = np.dot(X_train.T, I) / n
    sigma = np.dot((X_train - np.dot(I, mu.T)).T, (X_train - np.dot(I, mu.T))) / n
    print(mu.shape, sigma.shape)

    eigen_values, eigen_vectors = np.linalg.eigh(sigma)
    eigen_idx = np.argsort(eigen_values)[::-1]
    eigen_values_sorted, eigen_vectors_sorted = eigen_values[eigen_idx], eigen_vectors[:, eigen_idx]

    # A.3.a
    print(eigen_values_sorted[0], eigen_values_sorted[1], eigen_values_sorted[9], eigen_values_sorted[29], eigen_values_sorted[49])
    print('Summation of eigenvalues:', sum(eigen_values_sorted))

    # A.3.c
    current_eigen_values_sum = 0
    ks = []
    ratios = []
    train_losses = []
    test_losses = []

    def mse(Xhat, X):
        return np.linalg.norm(Xhat - X, ord=2) / len(X)

    for k in range(1, 101):
        X_train_reconstruct = np.dot(X_train - mu.T, np.dot(eigen_vectors_sorted[:, :k], eigen_vectors_sorted[:, :k].T)) + mu.reshape((784,))
        X_test_reconstruct = np.dot(X_test - mu.T, np.dot(eigen_vectors_sorted[:, :k], eigen_vectors_sorted[:, :k].T)) + mu.reshape((784,))
        train_losses.append(mse(X_train_reconstruct, X_train))
        test_losses.append(mse(X_test_reconstruct, X_test))
        ks.append(k)
        current_eigen_values_sum += eigen_values_sorted[k-1]
        ratios.append(1 - (current_eigen_values_sum/sum(eigen_values_sorted)))
        print(k, train_losses[-1], test_losses[-1])
    plt.figure(figsize=(10, 6))
    plt.plot(ks, train_losses, label='Train Reconstruction Error')
    plt.plot(ks, test_losses, label='Test Reconstruction Error')
    plt.xlabel('k')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.savefig('A3c_err.png')

    plt.figure(figsize=(10, 6))
    plt.plot(ks, ratios)
    plt.xlabel('k')
    plt.ylabel('Eigenvalue Ratio')
    plt.savefig('A3c_ratio.png')


    # A.3.d
    plt.figure(figsize=(10, 2))
    fig, axes = plt.subplots(1, 10)
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(eigen_vectors_sorted[:, i].reshape((28, 28)))
        ax.set_title('k = {}'.format(i + 1))
        ax.axis('off')
    plt.savefig('A3d.png')

    # A.3.e
    plt.figure(figsize=(10, 6))
    fig, axes = plt.subplots(3, 5)
    # Y_train[5] => 2, Y_train[13] => 6, Y_train[15]
    plot_labels = [(2, 5), (6, 13), (7, 15)]
    plot_ks = [-1, 5, 15, 40, 100]
    for nrow, (label, index) in enumerate(plot_labels):
        for ncol, k in enumerateimport matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mnist import MNIST

sns.set()

np.random.seed(1968990 + 20210531)


def load_dataset():
    mnist_data = MNIST('./mnist/')
    # X, Y = map(np.array, mnist_data.load_training())
    X_train, Y_train = map(np.array, mnist_data.load_training())
    X_test, Y_test = map(np.array, mnist_data.load_testing())
    # X_train, Y_train = X[:50000], Y[:50000]
    # X_test, Y_test = X[50000:], Y[50000:]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, Y_train, X_test, Y_test


def main():
    X_train, Y_train, X_test, Y_test = load_dataset()
    n, d = X_train.shape
    m, d = X_test.shape
    # print(X_train.shape)
    # print(X_test.shape)
    I = np.ones((n, 1))

    mu = np.dot(X_train.T, I) / n
    sigma = np.dot((X_train - np.dot(I, mu.T)).T, (X_train - np.dot(I, mu.T))) / n
    print(mu.shape, sigma.shape)

    eigen_values, eigen_vectors = np.linalg.eigh(sigma)
    eigen_idx = np.argsort(eigen_values)[::-1]
    eigen_values_sorted, eigen_vectors_sorted = eigen_values[eigen_idx], eigen_vectors[:, eigen_idx]

    # A.3.a
    print(eigen_values_sorted[0], eigen_values_sorted[1], eigen_values_sorted[9], eigen_values_sorted[29], eigen_values_sorted[49])
    print('Summation of eigenvalues:', sum(eigen_values_sorted))

    # A.3.c
    current_eigen_values_sum = 0
    ks = []
    ratios = []
    train_losses = []
    test_losses = []

    def mse(Xhat, X):
        return np.linalg.norm(Xhat - X, ord=2) / len(X)

    for k in range(1, 101):
        X_train_reconstruct = np.dot(X_train - mu.T, np.dot(eigen_vectors_sorted[:, :k], eigen_vectors_sorted[:, :k].T)) + mu.reshape((784,))
        X_test_reconstruct = np.dot(X_test - mu.T, np.dot(eigen_vectors_sorted[:, :k], eigen_vectors_sorted[:, :k].T)) + mu.reshape((784,))
        train_losses.append(mse(X_train_reconstruct, X_train))
        test_losses.append(mse(X_test_reconstruct, X_test))
        ks.append(k)
        current_eigen_values_sum += eigen_values_sorted[k-1]
        ratios.append(1 - (current_eigen_values_sum/sum(eigen_values_sorted)))
        print(k, train_losses[-1], test_losses[-1])
    plt.figure(figsize=(10, 6))
    plt.plot(ks, train_losses, label='Train Reconstruction Error')
    plt.plot(ks, test_losses, label='Test Reconstruction Error')
    plt.xlabel('k')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.savefig('A3c_err.png')

    plt.figure(figsize=(10, 6))
    plt.plot(ks, ratios)
    plt.xlabel('k')
    plt.ylabel('Eigenvalue Ratio')
    plt.savefig('A3c_ratio.png')


    # A.3.d
    plt.figure(figsize=(10, 2))
    fig, axes = plt.subplots(1, 10)
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(eigen_vectors_sorted[:, i].reshape((28, 28)))
        ax.set_title('k = {}'.format(i + 1))
        ax.axis('off')
    plt.savefig('A3d.png')

    # A.3.e
    plt.figure(figsize=(10, 6))
    fig, axes = plt.subplots(3, 5)
    # Y_train[5] => 2, Y_train[13] => 6, Y_train[15]
    plot_labels = [(2, 5), (6, 13), (7, 15)]
    plot_ks = [-1, 5, 15, 40, 100]
    for nrow, (label, index) in enumerate(plot_labels):
        for ncol, k in enumerate(plot_ks):
            if k == -1:
                axes[nrow, ncol].imshow(X_train[index].reshape(28, 28))
                axes[nrow, ncol].set_title('Original Image')
                axes[nrow, ncol].axis('off')
            else:
                reconstruct = (np.dot(X_train - mu.T, np.dot(eigen_vectors_sorted[:, :k], eigen_vectors_sorted[:, :k].T)) + mu.reshape((784,)))[index]
                axes[nrow, ncol].imshow(reconstruct.reshape(28, 28))
                axes[nrow, ncol].set_title('k = {}'.format(k))
                axes[nrow, ncol].axis('off')
    plt.savefig('A3e.png')


if __name__ == '__main__':
    main()(plot_ks):
            if k == -1:
                axes[nrow, ncol].imshow(X_train[index].reshape(28, 28))
                axes[nrow, ncol].set_title('Original Image')
                axes[nrow, ncol].axis('off')
            else:
                reconstruct = (np.dot(X_train - mu.T, np.dot(eigen_vectors_sorted[:, :k], eigen_vectors_sorted[:, :k].T)) + mu.reshape((784,)))[index]
                axes[nrow, ncol].imshow(reconstruct.reshape(28, 28))
                axes[nrow, ncol].set_title('k = {}'.format(k))
                axes[nrow, ncol].axis('off')
    plt.savefig('A3e.png')


if __name__ == '__main__':
    main()