if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from dezero import Variable

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

class HighgradTest(unittest.TestCase):
    def test_backward(self):
        x = Variable(np.array(2.0))
        y = f(x)
        y.backward(create_graph=True)
        # print(x.grad)
        self.assertEqual(x.grad.data, 24.0)

        gx = x.grad
        x.cleargrad()
        gx.backward()
        # print(x.grad)
        self.assertEqual(x.grad.data, 44.0)

unittest.main()