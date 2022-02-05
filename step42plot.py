import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.randn(100,1)
y = 5 + 2 * x + np.random.randn(100,1)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x, y, c='blue')
plt.show()

