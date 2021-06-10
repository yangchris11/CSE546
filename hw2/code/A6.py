import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mnist import MNIST

sns.set()

np.random.seed(1968990)


def load_dataset():
    mndata = MNIST('./data/')
    X_train, Y_train = map(np.array, mndata.load_training())
    X_test, Y_test = map(np.array, mndata.load_testing())
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, Y_train, X_test, Y_test


class LogisticRegression:
    def __init__(self, X, Y, X_test, Y_test, reg_lambda=1e-16, step=1e-2):
        self.X = X
        self.Y = Y
        self.X_test = X_test
        self.Y_test = Y_test
        self.reg_lambda = reg_lambda
        self.step = step
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.delta = 5e-4
        self.global_w_history = {}
        self.local_w_history = []
        self.local_b_history = []
        self.local_J_history = []
        self.local_J_test_history = []
        self.training_err = []
        self.test_err = []
        self.cur_iter = 0

    def grad_w(self, X, Y):
        return np.mean((((1/(1 + np.exp(-Y*(self.b + X.dot(self.w))))) - 1) * Y)[:, None] * X, axis=0) + \
               2 * self.reg_lambda * self.w

    def grad_b(self, X, Y):
        return np.mean(((1/(1 + np.exp(-Y*(self.b + X.dot(self.w))))) - 1) * Y, axis=0)

    def J(self, X, Y):
        return np.mean(np.log(1 + np.exp(-Y * (self.b + X.dot(self.w))))) + \
               self.reg_lambda * self.w.dot(self.w)

    def J_test(self, X, Y):
        return np.mean(np.log(1 + np.exp(-self.Y_test * (self.b + self.X_test.dot(self.w))))) + \
               self.reg_lambda * self.w.dot(self.w)

    def fit(self, batchsize):
        while self.cur_iter == 0 \
                or np.linalg.norm(self.w - self.local_w_history[-1]) > self.delta \
                and self.cur_iter < 10000: # max iteration

            batch = np.random.choice(self.X.shape[0], batchsize)
            X = self.X[batch]
            Y = self.Y[batch]

            self.cur_iter += 1
            self.local_w_history.append(np.copy(self.w))
            self.local_b_history.append(self.b)

            self.w -= self.step * self.grad_w(X, Y)
            self.b -= self.step * self.grad_b(X, Y)

            self.local_J_history.append(self.J(self.X, self.Y))
            self.local_J_test_history.append(self.J_test(self.X_test, self.Y_test))

            self.predict()

            if self.cur_iter % 10 == 0:
                print('Iter {}   | Training Loss: {:.5f} Training Acc: {:.2f} | '
                      'Training Loss: {:.5f} Training Acc: {:.2f} |'.format(
                        self.cur_iter,
                        self.local_J_history[-1],
                        100 * (1 - self.training_err[-1]),
                        self.local_J_test_history[-1],
                        100 * (1 - self.test_err[-1]))
                )

        self.local_w_history.append(np.copy(self.w))
        self.local_b_history.append(self.b)
        self.local_J_history.append(self.J(self.X, self.Y))
        self.local_J_test_history.append(self.J_test(self.X_test, self.Y_test))
        self.predict()

    def predict(self):
        Y_train_pred = np.sign(self.b + self.X.dot(self.w))
        Y_test_pred = np.sign(self.b + self.X_test.dot(self.w))
        self.training_err.append(1 - np.mean(self.Y == Y_train_pred))
        self.test_err.append(1 - np.mean(self.Y_test == Y_test_pred))


def main():
    print("Loading MNIST dataset from source")
    X_train, Y_train, X_test, Y_test = load_dataset()
    print(X_train.shape, X_test.shape)
    X_train_binary, Y_train_binary = [], []
    X_test_binary, Y_test_binary = [], []
    for data, label in zip(X_train, Y_train):
        if label == 7:
            X_train_binary.append(data)
            Y_train_binary.append(1)
        elif label == 2:
            X_train_binary.append(data)
            Y_train_binary.append(-1)
    for data, label in zip(X_test, Y_test):
        if label == 7:
            X_test_binary.append(data)
            Y_test_binary.append(1)
        elif label == 2:
            X_test_binary.append(data)
            Y_test_binary.append(-1)
    X_train_binary = np.asarray(X_train_binary)
    Y_train_binary = np.asarray(Y_train_binary)
    X_test_binary = np.asarray(X_test_binary)
    Y_test_binary = np.asarray(Y_test_binary)
    print(X_train_binary.shape)
    print(X_test_binary.shape)
    print("Finished loading binary MNIST dataset from source")

    model = LogisticRegression(X_train_binary, Y_train_binary, X_test_binary, Y_test_binary)
    model.fit(X_train_binary.shape[0])

    plt.figure(figsize=(10, 6))
    plt.plot(model.local_J_history,  label='Training Loss')
    plt.plot(model.local_J_test_history, label='Testing Loss')
    plt.xlabel('Iteration')
    plt.ylabel('$J(w,b)$')
    plt.legend()
    plt.savefig('A6b1.png')

    plt.figure(figsize=(10, 6))
    plt.plot(model.training_err, label='Training Error')
    plt.plot(model.test_err, label='Testing Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('A6b2.png')

    model = LogisticRegression(X_train_binary, Y_train_binary, X_test_binary, Y_test_binary)
    model.fit(1)

    plt.figure(figsize=(10, 6))
    plt.plot(model.local_J_history,  label='Training Loss')
    plt.plot(model.local_J_test_history, label='Testing Loss')
    plt.xlabel('Iteration')
    plt.ylabel('$J(w,b)$')
    plt.legend()
    plt.savefig('A6c1.png')

    plt.figure(figsize=(10, 6))
    plt.plot(model.training_err, label='Training Error')
    plt.plot(model.test_err, label='Testing Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('A6c2.png')

    model = LogisticRegression(X_train_binary, Y_train_binary, X_test_binary, Y_test_binary)
    model.fit(100)

    plt.figure(figsize=(10, 6))
    plt.plot(model.local_J_history,  label='Training Loss')
    plt.plot(model.local_J_test_history, label='Testing Loss')
    plt.xlabel('Iteration')
    plt.ylabel('$J(w,b)$')
    plt.legend()
    plt.savefig('A6d1.png')

    plt.figure(figsize=(10, 6))
    plt.plot(model.training_err, label='Training Error')
    plt.plot(model.test_err, label='Test Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('A6d2.png')


if __name__ == '__main__':
    main()