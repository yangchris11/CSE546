import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()


def main():
    plt.figure(figsize=(6, 6))
    x = np.linspace(-5.0, 5.0, 500)
    y = np.linspace(-5.0, 5.0, 500)
    X, Y = np.meshgrid(x, y)
    for i in np.arange(0, 4, 0.02):
        F = (X ** 0.5 + Y ** 0.5 ) ** 2 - i
        plt.contour(X, Y, F, [0])
        plt.contour(-X, Y, F, [0])
        plt.contour(-X, -Y, F, [0])
        plt.contour(X, -Y, F, [0])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('B2c.png')
    plt.show()
    plt.tight_layout()

if __name__ == '__main__':
    main()