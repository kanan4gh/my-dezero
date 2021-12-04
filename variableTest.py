#%% [markdown]
# ## 変数クラスの試作
#
# ここのコメント文はMarkdown形式で保存できます。

#%%
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable, Square, square, Exp, exp, as_array, numerical_diff

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

#%%
# 数値微分関数の定義(中心差分近似)
# f(x+h)-f(x-h)/2h 
# le-4=0.0001
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

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
assert y.creator.input == b
assert isinstance(y.creator.input.creator, Exp)
assert y.creator.input.creator.input == a
assert isinstance(y.creator.input.creator.input.creator, Square)
assert y.creator.input.creator.input.creator.input == x

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

