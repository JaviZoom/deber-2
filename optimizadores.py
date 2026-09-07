"""Optimizadores para actualizar parámetros `Value` a partir de sus gradientes.

Cada optimizador implementa `step(params)`, que lee `p.grad` (ya calculado por
`loss.backward()`) y actualiza `p.data`. El tamaño de batch (full-batch, SGD,
mini-batch) NO se decide acá, se decide en el bucle de entrenamiento (ver
`main_v2.py`): estos optimizadores solo definen la *regla de actualización*.
"""

import math


class GD:
    """Descenso por gradiente 'vanilla': p = p - lr * grad."""

    def __init__(self, lr=0.5):
        self.lr = lr

    def step(self, params):
        for p in params:
            p.data += -self.lr * p.grad


class Momentum:
    """GD + momentum: acumula una media móvil (velocidad) del gradiente."""

    def __init__(self, lr=0.2, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = None

    def step(self, params):
        if self.v is None:
            self.v = [0.0] * len(params)
        for i, p in enumerate(params):
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * p.grad
            p.data += -self.lr * self.v[i]


class Adam:
    """Adam: momentum (1er momento) + escalado adaptativo (2do momento)."""

    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)
        self.t += 1
        for i, p in enumerate(params):
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)  # corrección de sesgo
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data += -self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
