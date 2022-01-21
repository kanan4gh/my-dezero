import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)
    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad
    
    x.data -= gx.data / gx2.data

x = Variable(np.linspace(-2, 2, 400))
y = f(x)
logs = [y.data.flatten()]

plt.plot(x.data, y.data, label="y=x**4-2*x**2")
plt.legend(loc='lower right')
plt.show()



