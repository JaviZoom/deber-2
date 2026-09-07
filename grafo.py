import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def _layer_of(nodes, edges):
    """Capa de cada nodo = distancia más larga desde una hoja (sin hijos)."""
    children_of = {n: [] for n in nodes}
    for child, parent in edges:
        children_of[parent].append(child)

    cache = {}

    def layer(v):
        if v not in cache:
            kids = children_of[v]
            cache[v] = 0 if not kids else 1 + max(layer(k) for k in kids)
        return cache[v]

    return {n: layer(n) for n in nodes}


def draw_dot(root, filename="grafo.png"):
    """Dibuja el grafo computacional que termina en `root` (izquierda -> derecha) y lo guarda en `filename`."""
    nodes, edges = trace(root)
    layer_of = _layer_of(nodes, edges)

    layers = {}
    for n, layer in layer_of.items():
        layers.setdefault(layer, []).append(n)

    pos = {}
    for layer, layer_nodes in layers.items():
        layer_nodes.sort(key=lambda n: n.label or str(id(n)))
        offset = (len(layer_nodes) - 1) / 2.0
        for i, n in enumerate(layer_nodes):
            pos[n] = (layer * 3.2, offset - i)

    n_layers = max(layer_of.values()) + 1
    max_layer_size = max(len(v) for v in layers.values())
    fig, ax = plt.subplots(figsize=(max(6, 3.2 * n_layers), max(4, 1.3 * max_layer_size)))

    for n in nodes:
        x, y = pos[n]
        label = f"{n.label or '?'}\ndata={n.data:.4f}\ngrad={n.grad:.4f}"
        if n._op:
            label += f"\nop='{n._op}'"
        ax.add_patch(FancyBboxPatch((x - 1.0, y - 0.4), 2.0, 0.8,
                                     boxstyle="round,pad=0.05",
                                     facecolor="#eef3ff", edgecolor="#3355aa"))
        ax.text(x, y, label, ha="center", va="center", fontsize=7)

    for child, parent in edges:
        x1, y1 = pos[child]
        x2, y2 = pos[parent]
        ax.add_patch(FancyArrowPatch((x1 + 1.0, y1), (x2 - 1.0, y2),
                                      arrowstyle="-|>", mutation_scale=10,
                                      color="#555555", shrinkA=2, shrinkB=2))

    ax.set_xlim(-1.8, n_layers * 3.2)
    ax.set_ylim(-max_layer_size / 2.0 - 1, max_layer_size / 2.0 + 1)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    return filename
