if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from dezero import Variable, add, mul, using_config, no_grad

class OverloadTest(unittest.TestCase):
    def test_backward(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        c = Variable(np.array(1.0))

        y = add(mul(a,b), c)
        y.backward()

        self.assertEqual(y.data, 7.0)
        self.assertEqual(a.grad, 2.0)
        self.assertEqual(b.grad, 3.0) 

    def test_backward2(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        c = Variable(np.array(1.0))

        #y = add(mul(a,b), c)
        y = a * b + c
        y.backward()

        self.assertEqual(y.data, 7.0)
        self.assertEqual(a.grad, 2.0)
        self.assertEqual(b.grad, 3.0)    


unittest.main()