import numpy as np
from dezero import Variable
import dezero.functions as F
import matplotlib.pyplot as plt

# データセット
np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2 * np.pi * x) + np.random.rand(100,1)
xx, yy = x, y

# 重みの初期化
I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I,H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H,O))
b2 = Variable(np.zeros(O))

# 推論
def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y

lr = 0.2
iters = 10000

# 学習
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    if i % 500 == 0:
        print(loss)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x, y, c='blue')
ax.scatter(x, y_pred.data, c='red')
plt.show()