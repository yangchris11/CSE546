import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mnist import MNIST
from sklearn.utils import shuffle

sns.set()

def load_dataset():
    mndata = MNIST('./data/')
    X_train, Y_train = map(np.array, mndata.load_training())
    X_test, Y_test = map(np.array, mndata.load_testing())
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, Y_train, X_test, Y_test


def train(X, Y, reg_lambda=1e-16):
    I = np.eye(X.shape[1])
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
    W_hat = np.linalg.solve(X.T.dot(X) + reg_lambda * I, X.T.dot(Y))
    assert W_hat.shape == (X.shape[1], Y.shape[1])
    return W_hat


def predict(W, X):
    preds = np.argmax(W.T.dot(X.T), axis=0)
    assert preds.shape == (X.shape[0],)
    return preds

# Problem A.6
# def main():
#     print("Loading MNIST dataset from source")
#     X_train, Y_train, X_test, Y_test = load_dataset()
#     print("Finished loading MNIST dataset from source")
#
#     # # one-hot encoding the 10 classes in MNIST
#     # Y_train = np.eye(10)[Y_train]
#     # Y_test = np.eye(10)[Y_test]
#
#     W_hat = train(X_train, np.eye(10)[Y_train], 1e-4)
#
#     Y_train_preds = predict(W_hat, X_train)
#     Y_test_preds = predict(W_hat, X_test)
#
#     train_acc = np.count_nonzero(Y_train == Y_train_preds) / Y_train.shape[0]
#     print("Training Error:", 1-train_acc)
#     test_acc = np.count_nonzero(Y_test == Y_test_preds) / Y_test.shape[0]
#     print("Testing Error:", 1 - test_acc)

# Problem B.2.a
# def main():
#     print("Loading MNIST dataset from source")
#     X_train, Y_train, X_test, Y_test = load_dataset()
#     print("Finished loading MNIST dataset from source")
#
#     n = X_train.shape[0]
#
#     X, Y = shuffle(X_train, Y_train)
#     X_train = X[:int(0.8*n)]
#     Y_train = Y[:int(0.8*n)]
#     X_val = X[int(0.8*n):]
#     Y_val = Y[int(0.8*n):]
#
#     p_list = np.arange(200, 4000, 200)
#
#     train_err = []
#     val_err = []
#
#     for p in p_list:
#
#         G = np.random.normal(0, 0.1, (p, X_train.shape[1]))
#         b = np.random.uniform(0, 2*np.pi, (p,))
#         X_train_transform = np.cos(X_train.dot(G.T) + b)
#         W_hat = train(X_train_transform, np.eye(10)[Y_train], 1e-4)
#         X_val_transform = np.cos(X_val.dot(G.T) + b)
#         Y_train_preds = predict(W_hat, X_train_transform)
#         Y_val_preds = predict(W_hat, X_val_transform)
#
#         print("p=", p)
#         train_acc = np.count_nonzero(Y_train == Y_train_preds) / Y_train.shape[0]
#         train_err.append(1-train_acc)
#         print("Training Error:", 1-train_acc)
#         val_acc = np.count_nonzero(Y_val == Y_val_preds) / Y_val.shape[0]
#         val_err.append(1-val_acc)
#         print("Validation Error:", 1-val_acc)
#
#     plt.figure()
#     plt.plot(p_list, train_err, 'b--x', label='Training Error')
#     plt.plot(p_list, val_err, 'r--x', label='Validation Error')
#     plt.xlabel('$p$')
#     plt.ylabel('Error')
#     plt.legend()
#     plt.savefig('B-2-a.png')


# Problem B.2.b
def main():
    print("Loading MNIST dataset from source")
    X_train, Y_train, X_test, Y_test = load_dataset()
    print("Finished loading MNIST dataset from source")

    n = X_train.shape[0]

    X, Y = shuffle(X_train, Y_train)
    X_train = X[:int(0.8*n)]
    Y_train = Y[:int(0.8*n)]
    X_val = X[int(0.8*n):]
    Y_val = Y[int(0.8*n):]

    p_list = [4000]

    train_err = []
    val_err = []
    test_err = []

    for p in p_list:

        G = np.random.normal(0, 0.1, (p, X_train.shape[1]))
        b = np.random.uniform(0, 2*np.pi, (p,))

        X_train_transform = np.cos(X_train.dot(G.T) + b)
        X_val_transform = np.cos(X_val.dot(G.T) + b)
        X_test_transform = np.cos(X_test.dot(G.T) + b)

        W_hat = train(X_train_transform, np.eye(10)[Y_train], 1e-4)

        Y_train_preds = predict(W_hat, X_train_transform)
        Y_val_preds = predict(W_hat, X_val_transform)
        Y_test_preds = predict(W_hat, X_test_transform)

        print("p=", p)
        train_acc = np.count_nonzero(Y_train == Y_train_preds) / Y_train.shape[0]
        train_err.append(1-train_acc)
        print("Training Error:", 1-train_acc)
        val_acc = np.count_nonzero(Y_val == Y_val_preds) / Y_val.shape[0]
        val_err.append(1-val_acc)
        print("Validation Error:", 1-val_acc)
        test_acc = np.count_nonzero(Y_test == Y_test_preds) / Y_test.shape[0]
        test_err.append(1 - test_acc)
        print("Test Error:", 1 - test_acc)

    # plt.figure()
    # plt.plot(p_list, train_err, 'b--x', label='Training Error')
    # plt.plot(p_list, val_err, 'r--x', label='Validation Error')
    # plt.xlabel('$p$')
    # plt.ylabel('Error')
    # plt.legend()
    # plt.savefig('B-2-a.png')

if __name__ == '__main__':
    main()