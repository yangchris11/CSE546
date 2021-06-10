import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold, LeaveOneOut

sns.set()

np.random.seed(1968990 + 2021051)


# Polynomial kernel function
def k_poly(X, Z, d):
    return (1 + np.outer(X, Z)) ** d


# RBF kernel function
def k_rbf(X, Z, gamma):
    return np.exp(-gamma * np.subtract.outer(X, Z) ** 2)


class KRR:
    def __init__(self, type='poly', reg_lambda=1e-3, d=1, gamma=1):
        self.reg_lambda = reg_lambda
        if type == 'poly':
            self.kernel_fn = k_poly
            self.param = d
        if type == 'rbf':
            self.kernel_fn = k_rbf
            self.param = gamma
        self.X = None
        self.X_mean = None
        self.X_std = None
        self.alpha = None

    def fit(self, X, Y):
        self.X_mean, self.X_std = np.mean(X, axis=0), np.std(X, axis=0)
        self.X = (X - self.X_mean) / self.X_std
        # https://stackoverflow.com/questions/54771760/closed-form-ridge-regression
        # https://people.eecs.berkeley.edu/~wainwrig/stat241b/lec6.pdf
        self.alpha = np.linalg.solve(self.kernel_fn(self.X, self.X, self.param) + self.reg_lambda * np.eye(X.shape[0]), Y)

    def predict(self, Z):
        Z = (Z - self.X_mean) / self.X_std
        return np.dot(self.alpha, self.kernel_fn(self.X, Z, self.param))


# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html
def leave_one_out_cross_validation(X, Y, model):
    loo = LeaveOneOut()
    error = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        error.append(np.mean((Y_pred - Y_test) ** 2))
    return np.mean(error)


def k_fold_cross_validation(X, Y, model, k):
    if k == X.shape[0]:
        return leave_one_out_cross_validation(X, Y, model)
    kf = KFold(n_splits=k)
    error = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        error.append(np.mean((Y_pred - Y_test) ** 2))
    return np.mean(error)


# https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.cross_validation.Bootstrap.html
def bootstrap(X, Y, B, model):
    x = np.linspace(0, 1, 50)
    preds = []
    indices = np.arange(X.shape[0])

    for i in range(B):
        boot_index = np.random.choice(indices, size=X.shape[0], replace=True)
        X_boot, Y_boot = X[boot_index], Y[boot_index]
        model.fit(X_boot, Y_boot)
        preds.append(model.predict(x))

    return preds


def bootstrap_two_model(X, Y, B, poly_model, rbf_model, x):
    poly_preds, rbf_preds = [], []
    square_error = []
    indices = np.arange(X.shape[0])

    for i in range(B):
        boot_index = np.random.choice(indices, size=X.shape[0], replace=True)
        X_boot, Y_boot = X[boot_index], Y[boot_index]
        # model.fit(X_boot, Y_boot)
        poly_preds.append(poly_model.predict(X_boot))
        rbf_preds.append(rbf_model.predict(X_boot))
        square_error.append(np.mean((Y_boot - poly_preds[-1]) ** 2 - (Y_boot - rbf_preds[-1]) ** 2))

    print(np.percentile(square_error, 5, axis=0), np.percentile(square_error, 95, axis=0))


def main_poly(n, B):
    X = np.random.uniform(size=n)
    Y = 4 * np.sin(np.pi * X) * np.cos(6 * np.pi * X ** 2) + np.random.rand(n)

    model = KRR('poly')

    lambdas = 10. ** (-1 * np.arange(0, 10))
    ds = np.arange(2, 15)
    best_err = np.float('inf')
    best_lambda = None
    best_d = None

    for reg_lambda in lambdas:
        for d in ds:
            # print('Fitting Polynomial Kernel(lambda={}, d={})'.format(reg_lambda, d))
            model.reg_lambda = reg_lambda
            model.param = d
            error = leave_one_out_cross_validation(X, Y, model)
            if error < best_err:
                best_lambda = reg_lambda
                best_d = d
                best_err = error
    print('Best Hyper-Parameters for Polynomial Kernel: lambda={}, d={}'.format(best_lambda, best_d))

    # Plot
    model.reg_lambda = best_lambda
    model.param = best_d
    model.fit(X, Y)

    x = np.linspace(0.0, 1.0, 50)
    f_true = 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)
    f_hat = model.predict(x)

    plt.figure(figsize=(10, 6))
    plt.plot(X[np.argsort(X)], Y[np.argsort(X)], 'x', label='Original Data')
    plt.plot(x, f_true, label='True $f(x)$', color='green')
    plt.plot(x, f_hat, '--x', label='$\widehat f(x)$', color='orange')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.legend()
    plt.savefig('A3b_poly.png')
    plt.show()

    bootstrap_preds = bootstrap(X, Y, B, model)
    plt.figure(figsize=(10, 6))
    plt.plot(X[np.argsort(X)], Y[np.argsort(X)], 'x', label='Original Data')
    plt.plot(x, f_true, label='True $f(x)$', color='green')
    plt.plot(x, f_hat, '--x', label='$\widehat f(x)$', color='orange')
    plt.plot(x, np.percentile(bootstrap_preds, 5, axis=0), '--', color='red')
    plt.plot(x, np.percentile(bootstrap_preds, 95, axis=0), '--', color='red')
    plt.fill_between(x, np.percentile(bootstrap_preds, 5, axis=0), np.percentile(bootstrap_preds, 95, axis=0),
                     alpha=0.1, color="red")
    plt.ylim((-5, 5))
    plt.xlabel('x')
    plt.ylabel('f')
    plt.legend()
    plt.savefig('A3c_poly.png')
    plt.show()


def main_rbf(n, B):
    X = np.random.uniform(size=n)
    Y = 4 * np.sin(np.pi * X) * np.cos(6 * np.pi * X ** 2) + np.random.rand(n)

    model = KRR('rbf')

    lambdas = 10. ** (-1 * np.arange(0, 10))
    gammas = np.arange(2, 15, 0.5)
    best_err = np.float('inf')
    best_lambda = None
    best_gamma = None

    for reg_lambda in lambdas:
        for gamma in gammas:
            # print('Fitting Polynomial Kernel(lambda={}, gamma={})'.format(reg_lambda, gamma))
            model.reg_lambda = reg_lambda
            model.param = gamma
            error = leave_one_out_cross_validation(X, Y, model)
            if error < best_err:
                best_lambda = reg_lambda
                best_gamma = gamma
                best_err = error
    print('Best Hyper-Parameters for RBF kernel: lambda={}, gamma={}'.format(best_lambda, best_gamma))

    # Plot
    model.reg_lambda = best_lambda
    model.param = best_gamma
    model.fit(X, Y)

    x = np.linspace(0.0, 1.0, 50)
    f_true = 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)
    f_hat = model.predict(x)

    plt.figure(figsize=(10, 6))
    plt.plot(X[np.argsort(X)], Y[np.argsort(X)], 'x', label='Original Data')
    plt.plot(x, f_true, label='True $f(x)$', color='green')
    plt.plot(x, f_hat, '--x', label='$\widehat f(x)$', color='orange')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.legend()
    plt.savefig('A3b_rbf.png')

    bootstrap_preds = bootstrap(X, Y, B, model)
    plt.figure(figsize=(10, 6))
    plt.plot(X[np.argsort(X)], Y[np.argsort(X)], 'x', label='Original Data')
    plt.plot(x, f_true, label='True $f(x)$', color='green')
    plt.plot(x, f_hat, '--x', label='$\widehat f(x)$', color='orange')
    plt.plot(x, np.percentile(bootstrap_preds, 5, axis=0), '--', color='red')
    plt.plot(x, np.percentile(bootstrap_preds, 95, axis=0), '--', color='red')
    plt.fill_between(x, np.percentile(bootstrap_preds, 5, axis=0), np.percentile(bootstrap_preds, 95, axis=0),
                     alpha=0.1, color="red")
    plt.ylim((-5, 5))
    plt.xlabel('x')
    plt.ylabel('f')
    plt.legend()
    plt.savefig('A3c_rbf.png')
    plt.show()


def main_poly_kf(n, k, B):
    X = np.random.uniform(size=n)
    Y = 4 * np.sin(np.pi * X) * np.cos(6 * np.pi * X ** 2) + np.random.rand(n)

    model = KRR('poly')

    lambdas = 10. ** (-1 * np.arange(0, 10))
    ds = np.arange(2, 15)
    best_err = np.float('inf')
    best_lambda = None
    best_d = None

    for reg_lambda in lambdas:
        for d in ds:
            # print('Fitting Polynomial Kernel(lambda={}, d={})'.format(reg_lambda, d))
            model.reg_lambda = reg_lambda
            model.param = d
            error = k_fold_cross_validation(X, Y, model, k)
            if error < best_err:
                best_lambda = reg_lambda
                best_d = d
                best_err = error
    print('Best Hyper-Parameters for Polynomial Kernel: lambda={}, d={}'.format(best_lambda, best_d))

    # Plot
    model.reg_lambda = best_lambda
    model.param = best_d
    model.fit(X, Y)

    x = np.linspace(0.0, 1.0, 50)
    f_true = 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)
    f_hat = model.predict(x)

    plt.figure(figsize=(10, 6))
    plt.plot(X[np.argsort(X)], Y[np.argsort(X)], 'x', label='Original Data')
    plt.plot(x, f_true, label='True $f(x)$', color='green')
    plt.plot(x, f_hat, '--x', label='$\widehat f(x)$', color='orange')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.legend()
    plt.savefig('A3d_poly_1.png')

    bootstrap_preds = bootstrap(X, Y, B, model)
    plt.figure(figsize=(10, 6))
    plt.plot(X[np.argsort(X)], Y[np.argsort(X)], 'x', label='Original Data')
    plt.plot(x, f_true, label='True $f(x)$', color='green')
    plt.plot(x, f_hat, '--x', label='$\widehat f(x)$', color='orange')
    plt.plot(x, np.percentile(bootstrap_preds, 5, axis=0), '--', color='red')
    plt.plot(x, np.percentile(bootstrap_preds, 95, axis=0), '--', color='red')
    plt.fill_between(x, np.percentile(bootstrap_preds, 5, axis=0), np.percentile(bootstrap_preds, 95, axis=0),
                     alpha=0.1, color="red")
    plt.ylim((-5, 5))
    plt.xlabel('x')
    plt.ylabel('f')
    plt.legend()
    plt.savefig('A3d_poly_2.png')

    return model  # return best Poly model


def main_rbf_kf(n, B, k):
    X = np.random.uniform(size=n)
    Y = 4 * np.sin(np.pi * X) * np.cos(6 * np.pi * X ** 2) + np.random.rand(n)

    model = KRR('rbf')

    lambdas = 10. ** (-1 * np.arange(0, 10))
    gammas = np.arange(2, 15, 0.5)
    best_err = np.float('inf')
    best_lambda = None
    best_gamma = None

    for reg_lambda in lambdas:
        for gamma in gammas:
            # print('Fitting RBF Kernel(lambda={}, gamma={})'.format(reg_lambda, gamma))
            model.reg_lambda = reg_lambda
            model.param = gamma
            error = k_fold_cross_validation(X, Y, model, k)
            if error < best_err:
                best_lambda = reg_lambda
                best_gamma = gamma
                best_err = error
    print('Best Hyper-Parameters for RBF kernel: lambda={}, gamma={}'.format(best_lambda, best_gamma))

    # Plot
    model.reg_lambda = best_lambda
    model.param = best_gamma
    model.fit(X, Y)

    x = np.linspace(0.0, 1.0, 50)
    f_true = 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)
    f_hat = model.predict(x)

    plt.figure(figsize=(10, 6))
    plt.plot(X[np.argsort(X)], Y[np.argsort(X)], 'x', label='Original Data')
    plt.plot(x, f_true, label='True $f(x)$', color='green')
    plt.plot(x, f_hat, '--x', label='$\widehat f(x)$', color='orange')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.legend()
    plt.savefig('A3d_rbf_1.png')

    bootstrap_preds = bootstrap(X, Y, B, model)
    plt.figure(figsize=(10, 6))
    plt.plot(X[np.argsort(X)], Y[np.argsort(X)], 'x', label='Original Data')
    plt.plot(x, f_true, label='True $f(x)$', color='green')
    plt.plot(x, f_hat, '--x', label='$\widehat f(x)$', color='orange')
    plt.plot(x, np.percentile(bootstrap_preds, 5, axis=0), '--', color='red')
    plt.plot(x, np.percentile(bootstrap_preds, 95, axis=0), '--', color='red')
    plt.fill_between(x, np.percentile(bootstrap_preds, 5, axis=0), np.percentile(bootstrap_preds, 95, axis=0),
                     alpha=0.1, color="red")
    plt.ylim((-5, 5))
    plt.xlabel('x')
    plt.ylabel('f')
    plt.legend()
    plt.savefig('A3d_rbf_2.png')

    return model  # return best RBF model


def main_e(poly_model, rbf_model, n=1000, B=300):
    X = np.random.uniform(size=n)
    Y = 4 * np.sin(np.pi * X) * np.cos(6 * np.pi * X ** 2) + np.random.rand(n)

    poly_model.fit(X, Y)
    rbf_model.fit(X, Y)

    bootstrap_two_model(X, Y, B, poly_model, rbf_model, X)


if __name__ == '__main__':
    main_poly(n=50, B=300)
    main_rbf(n=50, B=300)
    poly_model = main_poly_kf(n=300, k=10, B=300)
    rbf_model = main_rbf_kf(n=300, k=10, B=300)
    main_e(poly_model, rbf_model)
