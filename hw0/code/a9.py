import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x1 = np.linspace(-5,5,100)
x2 = (x1 / 2.) - 1.0
plt.plot(x1, x2, '-r', label='$-x_1+2x_2+2=0$')

plt.xlim([-5, 5])

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()

plt.savefig('p9a.png')

plt.clf()

x1 = np.linspace(-5,5,100)
x2 = np.linspace(-5,5,100)
X1, X2 = np.meshgrid(x1,x2)
X3 = - X1 - X2

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(X1, X2, X3, color='r', label='$x_1+x_2+x_3=0$')
fake2Dline = matplotlib.lines.Line2D([0],[0], linestyle="none", c='r', marker = 'o')
ax.legend([fake2Dline], ['$x_1+x_2+x_3=0$'], numpoints = 1)

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')

fig.savefig('p9b.png')