import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

n = 40000
Z = np.random.randn(n)

plt.step(sorted(Z), np.arange(1, n+1) / float(n))
plt.xlim([-3, 3])
plt.ylim([0, 1])
plt.xlabel('Observations')
plt.ylabel('Probability')

plt.savefig('p12a.png')

K = [512, 64, 8, 1]

for k in K:
    Z = np.sum(np.sign(np.random.randn(n, k)) * np.sqrt(1./k), axis=1)
    plt.step(sorted(Z), np.arange(1, n + 1) / float(n))

plt.legend(['Gaussian', 512, 64, 8, 1])
plt.savefig('p12b.png')


