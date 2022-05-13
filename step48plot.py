import dezero
import matplotlib.pyplot as plt
import numpy as np

x, t = dezero.datasets.get_spiral(train=True)
print(x.shape)
print(t.shape)

print(x[10], t[10])
print(x[110], t[110])

x_0 = np.array([x_value for x_value, t_value in zip(x, t) if t_value == 0])
x_1 = np.array([x_value for x_value, t_value in zip(x, t) if t_value == 1])
x_2 = np.array([x_value for x_value, t_value in zip(x, t) if t_value == 2])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_0[:, 0], x_0[:, 1], marker="x", c='blue')
ax.scatter(x_1[:, 0], x_1[:, 1], marker="o", c='blue')
ax.scatter(x_2[:, 0], x_2[:, 1], marker="^", c='blue')
plt.show()

