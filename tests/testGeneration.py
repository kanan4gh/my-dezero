if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from dezero import Variable, add, square, numerical_diff

class GenerationTest(unittest.TestCase):

    def test_generation(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = add(square(a), square(a))
        y.backward()
    
        self.assertEqual(y.data, 32.0)
        self.assertEqual(x.grad, 64.0)

unittest.main()

