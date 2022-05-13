import numpy as np
from dezero import Variable
from dezero.models import MLP
import dezero.functions as F

#x = Variable(np.array([[1,2,3], [4,5,6]]))
#y = F.get_item(x, 1)
#print(y)

#y.backward()
#print(x.grad)

model = MLP((10,3))
x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])
y = model(x)
print(y)
loss = F.softmax_cross_entropy_simple(y, t)
print(loss)







