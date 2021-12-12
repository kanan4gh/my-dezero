if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from dezero import Variable, add, square, numerical_diff

class AddTest(unittest.TestCase):

    def test_call(self):
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        y = add(x0, x1)
        expected = 5
        self.assertEqual(y.data, expected)
    
    def test_multi(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))

        z = add(square(x), square(y))
        z.backward()
        self.assertEqual(z.data, 13)
        self.assertEqual(x.grad, 4)
        self.assertEqual(y.grad, 6)

    def test_multi2(self):
        x = Variable(np.array(3.0))
        y = add(add(x, x), x)

        y.backward()
        self.assertEqual(x.grad, 3.0)

    def test_multi3(self):
        x = Variable(np.array(3.0))
        y = add(x, x)
        y.backward()
        self.assertEqual(x.grad, 2.0)

        # 2回目の計算（同じxを使って、別の計算を行う）
        x.cleargrad()
        y = add(add(x, x), x)
        y.backward()
        self.assertEqual(x.grad, 3.0)
    

unittest.main()

