import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)

x = Variable(np.random.randn(1,2,3))
y = x.reshape((2,3))
yy = x.reshape(2,3)
print(y)
print(yy)

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.transpose(x)
y.backward()
print(x)
print(y)
print(x.grad)

x = Variable(np.random.randn(2,3))
y = x.T
y.backward()
print(x)
print(y)
print(x.grad)
