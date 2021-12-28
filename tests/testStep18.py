if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from dezero import Variable, add, square, using_config, no_grad

#
# メモリ使用量を減らすモードを追加する
#

class ReducemomoryTest(unittest.TestCase):
    def test_backward(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))
        t = add(x0, x1)
        y = add(x0, t)
        y.backward(retain_grad=True)

        self.assertEqual(y.grad, 1.0)
        self.assertEqual(t.grad, 1.0)
        self.assertEqual(x0.grad, 2.0)
        self.assertEqual(x1.grad, 1.0)

    def test_backward_without_retain(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))        
        t = add(x0, x1)
        y = add(x0, t)
        y.backward(retain_grad=False)
        self.assertEqual(y.grad, None)
        self.assertEqual(t.grad, None)
        self.assertEqual(x0.grad, 2.0)
        self.assertEqual(x1.grad, 1.0)

    def test_no_config_enable_backprop(self):
        # Config.enable_backprop = False
        with using_config("enable_backprop", False):
            x = Variable(np.ones((100, 100, 100)))
            y = square(square(square(x)))

    def test_using_config(self):
        with using_config("enable_backprop", False):
            x = Variable(np.array(2.0))
            y = square(x)

    def test_no_grad(self):
        with no_grad():
            x = Variable(np.array(2.0))
            y = square(x)
        
unittest.main()

