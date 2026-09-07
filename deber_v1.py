import os

import matplotlib.pyplot as plt

from grafo import draw_dot
from modelo import XS, YS, forward_xor, init_params
from motor import cross_entropy_compact

RESULTADOS_DIR = os.path.join("resultados", "deber_v1")
os.makedirs(RESULTADOS_DIR, exist_ok=True)

params = init_params(seed=0)


def main():
    # ENTRENAMIENTO
    lr = 0.5
    epochs = 3000
    loss_history = []

    for k in range(epochs):
        # 1) forward + loss
        sample_losses = [cross_entropy_compact(forward_xor(params, x1, x2), y_true)
                          for (x1, x2), y_true in zip(XS, YS)]
        loss = sum(sample_losses) * (1.0 / len(XS))  # promedio de la pérdida sobre el dataset
        loss.label = 'loss'
        loss_history.append(loss.data)

        # 2) backward (Backpropagation en los pesos y biases)
        for p in params:
            p.grad = 0.0
        loss.backward()

        # 3) update (Gradiente Descendiente)
        for p in params:
            p.data += -lr * p.grad

        if k % 300 == 0 or k == epochs - 1:
            print(f"epoch {k:04d} | loss = {loss.data:.6f}")

    # EVALUACIÓN
    print("\nPredicciones finales (sigmoid) y CE por muestra:")
    for (x1f, x2f), y_true in zip(XS, YS):
        L_sample = cross_entropy_compact(forward_xor(params, x1f, x2f), y_true)
        y_out = L_sample._prev.pop()
        pred = 1 if y_out.data >= 0.5 else 0
        print(f"({int(x1f)}, {int(x2f)}) -> out≈{y_out.data:.4f}  y_pred={pred}  y_true={int(y_true)}  CE≈{L_sample.data:.4f}")

    print("\nParámetros aprendidos:")
    for p in params:
        print(f"{p.label} = {p.data:.6f}")

    # Curva de pérdida
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history)
    plt.xlabel("epoch")
    plt.ylabel("loss (BCE)")
    plt.title("Entrenamiento XOR - MLP 2-2-1")
    plt.tight_layout()
    loss_path = os.path.join(RESULTADOS_DIR, "loss_xor.png")
    plt.savefig(loss_path)
    print(f"\nCurva de pérdida guardada en {loss_path}")

    # Grafo computacional de un ejemplo, en Colab se ve al vuelo bloque por
    # bloque; aquí guardamos ambas "fotos" como archivos PNG
    for p in params:
        p.grad = 0.0
    L_vis = cross_entropy_compact(forward_xor(params, 1.0, 0.0), 1.0)

    # 1) justo después del forward: todos los grad en 0.0000
    forward_path = os.path.join(RESULTADOS_DIR, "grafo_xor_forward.png")
    draw_dot(L_vis, forward_path)
    print(f"Grafo (forward, antes de backward) guardado en {forward_path}")

    # 2) después de backward: mismos nodos, ahora con los gradientes calculados
    L_vis.backward()
    backward_path = os.path.join(RESULTADOS_DIR, "grafo_xor_backward.png")
    draw_dot(L_vis, backward_path)
    print(f"Grafo (después de backward) guardado en {backward_path}")


if __name__ == "__main__":
    main()
