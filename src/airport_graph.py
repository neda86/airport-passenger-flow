"""Canonical airport graph definition.

This is the SINGLE source of truth for node names, node order, and the
adjacency matrix. Both feature engineering and the models import from here,
which fixes the node-ordering bug (feature tensor rows and adjacency rows
previously came from two unrelated orderings).
"""
import numpy as np
import networkx as nx

CHECKPOINTS = ["Arrival", "Check-in", "Security", "Boarding"]
NUM_GATES = 20  # the paper says Gates 1-20; the simulator must assign all of them
GATES = [f"Gate {i}" for i in range(1, NUM_GATES + 1)]

# Canonical node order — index i here is row/col i of A and node-slot i of
# every feature/target tensor.
NODES = CHECKPOINTS + GATES
NODE_INDEX = {name: i for i, name in enumerate(NODES)}
NUM_NODES = len(NODES)  # 24


def build_graph() -> nx.DiGraph:
    edges = [
        ("Arrival", "Check-in"),
        ("Check-in", "Security"),
        ("Security", "Boarding"),
    ]
    # Two concourses branching off Boarding, gates connected sequentially.
    edges.append(("Boarding", "Gate 1"))
    for i in range(1, 10):
        edges.append((f"Gate {i}", f"Gate {i + 1}"))
    edges.append(("Boarding", "Gate 11"))
    for i in range(11, 20):
        edges.append((f"Gate {i}", f"Gate {i + 1}"))

    G = nx.DiGraph()
    G.add_nodes_from(NODES)
    G.add_edges_from(edges)
    return G


def adjacency() -> np.ndarray:
    """Raw unweighted adjacency in canonical node order (Eq. 1 of the paper)."""
    return nx.to_numpy_array(build_graph(), nodelist=NODES, dtype=float)


def normalized_adjacency() -> np.ndarray:
    """Symmetrically normalized adjacency with self-loops (Eq. 3-4 of the paper):
    A_norm = D^{-1/2} (A + I) D^{-1/2}

    The directed A is symmetrized first: GCN message passing over a DAG with
    in-degree-0 source nodes is ill-conditioned, and physically passengers'
    states at adjacent checkpoints inform each other in both directions.
    """
    A = adjacency()
    A = np.maximum(A, A.T)  # symmetrize
    A_hat = A + np.eye(A.shape[0])
    d = A_hat.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


if __name__ == "__main__":
    A = adjacency()
    An = normalized_adjacency()
    print(f"{NUM_NODES} nodes: {NODES}")
    print("A shape:", A.shape, "| edges:", int(A.sum()))
    print("A_norm symmetric:", np.allclose(An, An.T))
    print("A_norm row of 'Boarding':", np.round(An[NODE_INDEX["Boarding"]], 3))
