from dezero import Variable
import dezero.functions as F
import numpy as np

# 行列計算のテスト

x = Variable(np.random.randn(2, 3))
W = Variable(np.random.randn(3, 4))
y = F.matmul(x, W)
y.backward()

print(x.grad.shape)
print(W.grad.shape)

a = Variable(np.array([[1,2], [3,4]]))
b = Variable(np.array([[5,6], [7,8]]))
c = F.matmul(a, b)
print(c.data)




