import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from lasso import Lasso

sns.set()


def count(w_true, w_train, k):
    nonzero_cnt = np.sum(abs(w_train) > 0)
    fdr = np.sum(abs(w_train[k:]) > 0) / nonzero_cnt
    tpr = np.sum(abs(w_train[:k]) > 0) / k
    return nonzero_cnt, fdr, tpr

def main():
    np.random.seed(1968990)

    n, d, k, sigma = 500, 1000, 100, 1
    m = 20
    w = np.zeros((d, ))
    for j in range(1, k+1):
        w[j-1] = j/k
    X = np.random.normal(size=(n, d))
    y = X.dot(w) + np.random.normal(size=(n,))
    # print(X.shape, y.shape, w.shape)

    reg_lambda = max(2*np.sum(X*(y-np.mean(y))[:, None], axis=0))
    # reg_lambda = max(2*np.sum(X.T*(y-np.mean(y)), axis=0))
    print('Max lambda: ', reg_lambda)

    model = Lasso(X, y, reg_lambda)

    lambdas = []
    nonzeros = []
    fdrs = []
    tprs = []
    for _ in range(m):
        model.fit()
        lambdas.append(reg_lambda)

        nz, fdr, tpr = count(w, model.w, k)
        nonzeros.append(nz)
        fdrs.append(fdr)
        tprs.append(tpr)

        reg_lambda /= 2
        model.reset(reg_lambda)

    plt.figure(figsize=(10,6))
    plt.plot(lambdas, nonzeros, '--x')
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('Number of Non-zeros')
    plt.savefig('A4a.png')
    plt.tight_layout()

    plt.figure(figsize=(10,6))
    plt.plot(fdrs, tprs, 'rx')
    plt.xlabel('False Discovery Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('A4b.png')
    plt.tight_layout()

if __name__ == '__main__':
    main()