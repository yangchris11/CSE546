"""
    TEST SCRIPT FOR POLYNOMIAL REGRESSION 1
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from polyreg import PolynomialRegression

cmap = matplotlib.cm.get_cmap('winter')

import seaborn as sns

sns.set()

if __name__ == "__main__":
    '''
        Main function to test polynomial regression
    '''

    # load the data
    filePath = "data/polydata.dat"
    file = open(filePath,'r')
    allData = np.loadtxt(file, delimiter=',')
    X = allData[:, [0]]
    y = allData[:, [1]]

    # regression with degree = d
    d = 8
    model = PolynomialRegression(degree=d, reg_lambda=0)
    model.fit(X, y)

    # output predictions
    xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
    ypoints = model.predict(xpoints)

    # plot curve
    plt.figure()
    plt.plot(X, y, 'rx')
    plt.plot(xpoints, ypoints, color=cmap(0), label='$\lambda$=0')

    model = PolynomialRegression(degree=d, reg_lambda=1e-4)
    model.fit(X, y)
    ypoints = model.predict(xpoints)
    plt.plot(xpoints, ypoints, color=cmap(0.35), label='$\lambda$=0.0001')

    model = PolynomialRegression(degree=d, reg_lambda=1e-2)
    model.fit(X, y)
    ypoints = model.predict(xpoints)
    plt.plot(xpoints, ypoints, color=cmap(0.7), label='$\lambda$=0.01')

    model = PolynomialRegression(degree=d, reg_lambda=1)
    model.fit(X, y)
    ypoints = model.predict(xpoints)
    plt.plot(xpoints, ypoints, color=cmap(1.2), label='$\lambda$=1')

    plt.title('PolyRegression with d = ' + str(d))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig('A_4.png')
    plt.show()
