import numpy as np

class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        # None以外のパラメータをリストにまとめる
        params = [p for p in self.target.params() if p.grad is not None]

        # 前処理（オプション）
        for f in self.hooks:
            f(params)

        # パラメータ更新
        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)

# 確率的勾配降下法
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data

# Momentum
class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momuntum=0.9):
        super().__init__()
        self.lr = lr
        self.momuntum = momuntum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momuntum
        v -= self.lr * param.grad.data
        param.data += v






