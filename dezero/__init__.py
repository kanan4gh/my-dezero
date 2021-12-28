# ===============================================================
# step23.pyからstep32.pyまではsimple_coreを利用
is_simple_core = True
# ===============================================================

if is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import no_grad
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_variable

    from dezero.core_simple import Add
    from dezero.core_simple import add
    from dezero.core_simple import Mul
    from dezero.core_simple import mul
    from dezero.core_simple import Square
    from dezero.core_simple import square
    from dezero.core_simple import Exp
    from dezero.core_simple import exp
    from dezero.core_simple import numerical_diff

else:
    from dezero.core_simple import Variable, setup_variable

setup_variable()
__version__ = '0.0.13'

