import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

n = 256
sigma = 1
M = [1, 2, 4, 8, 16, 32]
eps = np.random.normal(0, sigma, n)

X = np.arange(1, n+1) / n
Y = 4 * np.sin(np.pi * X) * np.cos(6 * np.pi * X * X) + eps
Y_true = 4 * np.sin(np.pi * X) * np.cos(6 * np.pi * X * X)

# average empirical error
average_empirical_error = []
for m in M:
    f_m = np.array([np.mean(Y[j*m-m:j*m]) for j in np.arange(1, n//m + 1)])
    f_hat = np.array([f_m[int((i-1)//m)] for i in np.arange(1, n+1)])
    err = np.mean((f_hat - Y_true)**2)
    average_empirical_error.append(err)

# average bias-squared error
average_bias_squared_error = []
for m in M:
    f_j = np.array([np.mean(Y_true[j * m - m:j * m]) for j in np.arange(1, n // m + 1)])
    f_hat = np.array([f_j[int((i - 1) // m)] for i in np.arange(1, n + 1)])
    err = np.mean((f_hat - Y_true) ** 2)
    average_bias_squared_error.append(err)

# average variance error
average_variance_error = []
for m in M:
    average_variance_error.append(sigma**2/m)

plt.plot(M, average_empirical_error, 'x--', label='Average Empirical Error')
plt.plot(M, average_bias_squared_error, 'x--', label='Average Bias-Squared Error')
plt.plot(M, average_variance_error, 'x--', label='Average Variance Error')
plt.plot(M, np.array(average_bias_squared_error) + np.array(average_variance_error), 'x--', label='Average Error')
plt.legend()
plt.xlabel('$m$')
plt.ylabel('Error')
plt.savefig('B-1-4.png')