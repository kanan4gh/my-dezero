#%% [markdown]
# ## 変数クラスの試作
#
# ここのコメント文はMarkdown形式で保存できます。

#%%
# 
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable, Square, square, Exp, exp, as_array, Add, add, numerical_diff

#%%
# numpy.arrayで変数オブジェクトを作成する
data = np.array(1.0)
x = Variable(data)
print(x.data)

# %%
# 変数オブジェクトxに新しいデータを代入する
x.data = np.array(2.0)
print(x.data)

# %%
# numpy多次元配列
x1 = np.array(1)
print(x1.ndim)

x2 = np.array([1,2,3])
print(x2.ndim)

x3 = np.array([[1,2,3],
               [4,5,6]])
print(x3.ndim)

# %%
#
x = Variable(np.array(10))
y = square(x)
print(type(y))
print(y.data)

f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f,x)
print(dy)

# %%
# 関数連鎖による順伝搬と逆伝搬の例
x = Variable(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b)
print(y.data)

#逆向きに計算グラフのノードを辿る
assert isinstance(y.creator, Square)
assert y.creator.inputs[0] == b
assert isinstance(y.creator.inputs[0].creator, Exp)
assert y.creator.inputs[0].creator.inputs[0] == a
assert isinstance(y.creator.inputs[0].creator.inputs[0].creator, Square)
assert y.creator.inputs[0].creator.inputs[0].creator.inputs[0] == x

y.backward()
print(x.grad)

# 数値微分との比較
def f(x):
    return square(exp(square(x)))

x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print(dy)

x = Variable(None)
try:
    x = Variable(1.0) # NG:エラーが発生
except TypeError as e:
    print(e)

#%% 変数を繰り返す使う
#
x = Variable(np.array(3.0))
y = add(x, x)
print('y', y.data)

y.backward()
print('x.grad', x.grad)

# %% インプレース演算の検証
#
x = np.array(1.0)
print(id(x))
y = x
print(id(y))
y += x
print(x, y, id(x), id(y))

x = np.array(1.0)
print(id(x))
y = np.array(2.0)
print(id(y))
y = y +x
print(y, id(x), id(y))

# %%
