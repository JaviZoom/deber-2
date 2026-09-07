"""
DEBER 2 (versión 2): comparación de optimizadores para entrenar el mismo
MLP 2-2-1 sobre la compuerta XOR.

En vez de quedarnos solo con el descenso por gradiente "vanilla" (full-batch),
comparamos 5 variantes, separando dos ideas independientes:

  - Cómo se arma cada actualización (batch_size): Batch GD (todo el dataset),
    SGD (una muestra a la vez) y Mini-batch GD (de a 2 muestras).
  - Qué regla de actualización usa el gradiente (optimizer): GD "vanilla",
    Momentum y Adam.

Todas parten de los mismos pesos iniciales (misma semilla) para que la
comparación sea justa.
"""

import os
import random

import matplotlib.pyplot as plt

from modelo import XS, YS, forward_xor, init_params
from motor import cross_entropy_compact
from optimizadores import GD, Adam, Momentum

RESULTADOS_DIR = os.path.join("resultados", "deber_v2")
os.makedirs(RESULTADOS_DIR, exist_ok=True)

SEED = 0
EPOCHS = 1000
LOSS_OK = 0.05  # umbral arbitrario, solo para comparar qué tan rápido baja cada curva


def dataset_loss(params):
    """Pérdida BCE promedio sobre las 4 muestras de XOR (para comparar curvas)."""
    losses = [cross_entropy_compact(forward_xor(params, x1, x2), y)
              for (x1, x2), y in zip(XS, YS)]
    return (sum(losses) * (1.0 / len(XS))).data


def accuracy_of(params):
    hits = sum(1 for (x1, x2), y_true in zip(XS, YS)
               if (forward_xor(params, x1, x2).data >= 0.5) == bool(y_true))
    return hits / len(XS)


def train(optimizer, batch_size, epochs=EPOCHS, seed=SEED):
    params = init_params(seed=seed)
    indices = list(range(len(XS)))
    history = []

    for epoch in range(epochs):
        random.seed(seed * 10_000 + epoch)  # orden de batches reproducible
        random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            batch_losses = [cross_entropy_compact(forward_xor(params, XS[i][0], XS[i][1]), YS[i])
                             for i in batch]
            loss = sum(batch_losses) * (1.0 / len(batch))

            for p in params:
                p.grad = 0.0
            loss.backward()
            optimizer.step(params)

        history.append(dataset_loss(params))

    return params, history


def evaluate(name, params, history):
    epoch_ok = next((e for e, loss in enumerate(history) if loss < LOSS_OK), None)
    epoch_ok_str = f"{epoch_ok:04d}" if epoch_ok is not None else f">{EPOCHS} (no llegó)"

    print(f"{name:14s} | loss final = {history[-1]:.6f} | accuracy = {accuracy_of(params) * 100:.0f}% "
          f"| primera época con loss < {LOSS_OK}: {epoch_ok_str}")


def main():
    """Entrena XOR con 5 optimizadores distintos y compara sus curvas de pérdida."""
    experiments = [
        ("Batch GD",      GD(lr=0.5),                 len(XS)),  # full-batch, 1 update/epoch
        ("SGD",           GD(lr=0.3),                  1),        # 1 muestra, 4 updates/epoch
        ("Mini-batch GD", GD(lr=0.5),                  2),        # 2 muestras, 2 updates/epoch
        ("Momentum GD",   Momentum(lr=0.8, beta=0.9),  len(XS)),  # full-batch + momentum
        ("Adam",          Adam(lr=0.1),                len(XS)),  # full-batch + Adam
    ]

    results = {}
    for name, optimizer, batch_size in experiments:
        params, history = train(optimizer, batch_size)
        results[name] = (params, history)

    print(f"\nComparación tras {EPOCHS} épocas (mismo seed, misma arquitectura 2-2-1):\n")
    for name, (params, history) in results.items():
        evaluate(name, params, history)

    # Curvas de pérdida superpuestas
    plt.figure(figsize=(7, 5))
    for name, (_params, history) in results.items():
        plt.plot(history, label=name)
    plt.xlabel("epoch")
    plt.ylabel("loss (BCE)")
    plt.yscale("log")
    plt.title("XOR - comparación de optimizadores (2-2-1)")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(RESULTADOS_DIR, "loss_comparacion_optimizadores.png")
    plt.savefig(plot_path)
    print(f"\nCurvas comparadas guardadas en {plot_path}")


if __name__ == "__main__":
    main()
