"""Arquitectura del MLP 2-2-1 (XOR) compartida entre las versiones del Deber 2.

Capa oculta (1): 2 neuronas sigmoide
  z1 = w1_11*x1 + w1_12*x2 + b1_1 ; y1 = sigmoid(z1)
  z2 = w1_21*x1 + w1_22*x2 + b1_2 ; y2 = sigmoid(z2)
Capa de salida (2): 1 neurona sigmoide
  z3 = w2_11*y1 + w2_21*y2 + b2_1 ; y_pred = sigmoid(z3)
"""

import random

from motor import Value

PARAM_LABELS = ['w1_11', 'w1_12', 'b1_1', 'w1_21', 'w1_22', 'b1_2', 'w2_11', 'w2_21', 'b2_1']

# DATASET XOR
XS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
YS = [0.0, 1.0, 1.0, 0.0]


def init_params(seed=0):
    """Inicializa los 9 parámetros del MLP con la misma semilla (comparaciones justas)."""
    random.seed(seed)
    return [Value(random.uniform(-1, 1), label=label) for label in PARAM_LABELS]


def forward_xor(params, x1_float, x2_float):
    w1_11, w1_12, b1_1, w1_21, w1_22, b1_2, w2_11, w2_21, b2_1 = params

    x1 = Value(x1_float, label='x1')
    x2 = Value(x2_float, label='x2')

    z1 = x1 * w1_11 + x2 * w1_12 + b1_1; z1.label = 'z1'
    y1 = z1.sigmoid(); y1.label = 'y1'

    z2 = x1 * w1_21 + x2 * w1_22 + b1_2; z2.label = 'z2'
    y2 = z2.sigmoid(); y2.label = 'y2'

    z3 = y1 * w2_11 + y2 * w2_21 + b2_1; z3.label = 'z3'
    y_pred = z3.sigmoid(); y_pred.label = 'y_pred'
    return y_pred
