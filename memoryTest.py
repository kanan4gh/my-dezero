if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from memory_profiler import profile
from dezero import Variable, square

@profile
def profiler_base():
    pass

@profile
def profiler_target_func():
    for i in range(10):
        x = Variable(np.random.randn(10000))
        y = square(square(square(x)))

def main():
    profiler_base()
    profiler_target_func()

if __name__ == "__main__":
    main()
