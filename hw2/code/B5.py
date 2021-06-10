import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import Ridge


sns.set()


np.random.seed(1)

def main():
    n = 20000
    d = 10000
    canonical = np.eye(d)
    X = []
    eps = np.random.normal(0, 1, size=(n,))

    for i in range(1, n+1):
        X.append(canonical[i % d] * np.sqrt((i % d) + 1))

    X = np.asarray(X)
    Y = eps

    reg = Ridge(alpha=0)
    reg.fit(X, Y)

    xtx = np.linalg.inv(np.dot(X.T, X))

    beta = []
    upper = []
    lower = []
    cnt = 0
    for j in range(d):
        beta.append(reg.coef_[j])
        upper.append(np.sqrt(2*xtx[j][j]*np.log(2/0.05)))
        lower.append(-np.sqrt(2*xtx[j][j]*np.log(2/0.05)))
        if reg.coef_[j] >= np.sqrt(2*xtx[j][j]*np.log(2/0.05)) or reg.coef_[j] <= -np.sqrt(2*xtx[j][j]*np.log(2/0.05)):
            cnt += 1

    print(cnt)

    dd = np.arange(d)
    plt.figure(figsize=(10, 6))
    plt.scatter(dd, beta, marker='x')
    plt.plot(upper, 'r')
    plt.plot(lower, 'r')
    plt.xlabel('$j$')
    plt.ylabel(r"$\beta_j$")
    plt.savefig('B5c.png')
    plt.show()

    print(max(upper))


if __name__ == '__main__':
    main()