import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lasso import Lasso

sns.set()


def count(w_train):
    nonzero_cnt = np.sum(abs(w_train) > 0)
    return nonzero_cnt

def main():
    m = 20
    df_train = pd.read_table("crime-train.txt")
    df_test = pd.read_table("crime-test.txt")

    X = df_train.drop('ViolentCrimesPerPop', axis=1)
    y = df_train['ViolentCrimesPerPop']
    # print(X.shape, y.shape)

    X_test = df_test.drop('ViolentCrimesPerPop', axis=1)
    y_test = df_test['ViolentCrimesPerPop']

    reg_lambda = max(2 * np.sum(X.T * (y - np.mean(y)), axis=0))
    print('Max lambda: ', reg_lambda)

    model = Lasso(X.values, y.values, reg_lambda)

    lambdas = []
    nonzeros = []
    mse_train = []
    mse_test = []

    for _ in range(m):
        model.fit()
        lambdas.append(reg_lambda)
        nonzeros.append(np.sum(abs(model.w) > 0))
        y_train_preds = model.predict(X.values)
        y_test_preds = model.predict(X_test.values)

        mse_train.append(np.mean((y.values-y_train_preds)**2))
        mse_test.append(np.mean((y_test.values-y_test_preds)**2))

        reg_lambda /= 2
        model.reset(reg_lambda)

    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, nonzeros, '--x')
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('Number of Non-zeros')
    plt.savefig('A5c.png')
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    vis_feat_name = ['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize']
    for name in vis_feat_name:
        idx = X.columns.get_loc(name)
        w_path = []
        for key, val in model.global_w_history.items():
            w_path.append(val[idx])
        plt.plot(lambdas, w_path, '--x', label=name)
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('Regularization Paths')
    plt.legend()
    plt.savefig('A5d.png')
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, mse_train, '--x', label='Train')
    plt.plot(lambdas, mse_test, '--x', label='Test')
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.savefig('A5e.png')
    plt.tight_layout()

    model.reset(30)
    model.fit()
    print('Largest Lasso coefficient: ', X.columns[np.argmax(model.w)], max(model.w))
    print('Smallest Lasso coefficient: ', X.columns[np.argmin(model.w)], min(model.w))


if __name__ == '__main__':
    main()