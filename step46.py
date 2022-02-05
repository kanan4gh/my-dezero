import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iter = 10000
hidden_size = 10

model = MLP((hidden_size, 1))
# optimizer = optimizers.SGD(lr).setup(model)
optimizer = optimizers.MomentumSGD(lr).setup(model)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()
    
    if i % 1000 == 0:
        print(loss)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x, y, c='blue')
ax.scatter(x, y_pred.data, c='red')
plt.show()

