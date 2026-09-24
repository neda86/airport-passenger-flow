"""MCO-inspired terminal layout (Orlando International, North Terminal).

Structure modelled (simplified, not an exact replica):

  Arrival ──> Check-in ──┬──> Security West ──┐        ┌──> Airside 1 ──> Gates 1-6
     │                   │                    ├─ APM ──┼──> Airside 2 ──> Gates 7-12
     └── (no checked bag)┴──> Security East ──┘        ├──> Airside 3 ──> Gates 13-18
                                                       └──> Airside 4 ──> Gates 19-24

Key differences vs the linear layout:
  * TWO parallel security checkpoints (West / East) instead of one.
  * Passengers without checked bags skip Check-in (online check-in).
  * Automated people mover (APM) from each checkpoint to four airside
    concourses; each airside is a hub node for its gate cluster.
  * 32 nodes: 8 terminal-process nodes + 24 gates.

Exposes the same API as airport_graph_linear so every downstream script
works unchanged: NODES, NODE_INDEX, NUM_NODES, CHECKPOINTS, GATES, NUM_GATES,
adjacency(), normalized_adjacency(), plus SCHED_DEP_NODES, GATE_HUB and
node_flow_events(df).
"""
import numpy as np
import networkx as nx

LAYOUT_NAME = "mco"

NUM_AIRSIDES = 4
GATES_PER_AIRSIDE = 6
NUM_GATES = NUM_AIRSIDES * GATES_PER_AIRSIDE          # 24
SECURITY = ["Security West", "Security East"]
AIRSIDES = [f"Airside {k}" for k in range(1, NUM_AIRSIDES + 1)]
CHECKPOINTS = ["Arrival", "Check-in"] + SECURITY + AIRSIDES   # 8 process nodes
GATES = [f"Gate {i}" for i in range(1, NUM_GATES + 1)]

NODES = CHECKPOINTS + GATES
NODE_INDEX = {name: i for i, name in enumerate(NODES)}
NUM_NODES = len(NODES)                                  # 32

SCHED_DEP_NODES = ["Arrival", "Check-in"] + SECURITY


def airside_of_gate(g: int) -> int:
    """1-based airside index for 1-based gate number."""
    return (g - 1) // GATES_PER_AIRSIDE + 1


GATE_HUB = {f"Gate {g}": f"Airside {airside_of_gate(g)}" for g in range(1, NUM_GATES + 1)}


def build_graph() -> nx.DiGraph:
    edges = [("Arrival", "Check-in")]
    for s in SECURITY:
        edges.append(("Check-in", s))
        edges.append(("Arrival", s))            # bag-less passengers skip check-in
        for a in AIRSIDES:
            edges.append((s, a))                # APM links
    for k, a in enumerate(AIRSIDES, start=1):
        gates = [f"Gate {g}" for g in range(1, NUM_GATES + 1) if airside_of_gate(g) == k]
        edges.append((a, gates[0]))
        for g1, g2 in zip(gates[:-1], gates[1:]):
            edges.append((g1, g2))              # pier: gates in sequence
    G = nx.DiGraph()
    G.add_nodes_from(NODES)
    G.add_edges_from(edges)
    return G


def adjacency() -> np.ndarray:
    return nx.to_numpy_array(build_graph(), nodelist=NODES, dtype=float)


def normalized_adjacency() -> np.ndarray:
    A = adjacency()
    A = np.maximum(A, A.T)
    A_hat = A + np.eye(A.shape[0])
    d = A_hat.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


def node_flow_events(df):
    """One row per (node, flow-type) event for the MCO layout.

    Expects passenger columns: Arrival Time, Check-in Start/End (NaT if the
    passenger skipped check-in), Checkpoint ('West'/'East'), Security Queue
    Start/End, Airside (1-4), Airside Arrival, Boarding Time, Gate.
    """
    import pandas as pd
    for col in ["Arrival Time", "Check-in Start", "Check-in End",
                "Security Queue Start", "Security Queue End",
                "Airside Arrival", "Boarding Time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    recs = []

    def add(node, start, end, sub):
        sub = sub[[start, end]].dropna()
        recs.append(pd.DataFrame({"Node": node, "Time": sub[start], "Type": "in"}))
        recs.append(pd.DataFrame({"Node": node, "Time": sub[end], "Type": "out"}))

    # Arrival: leaves either to check-in or (bag-less) straight to security.
    leave = df["Check-in Start"].fillna(df["Security Queue Start"])
    arr = pd.DataFrame({"a": df["Arrival Time"], "b": leave}).dropna()
    recs.append(pd.DataFrame({"Node": "Arrival", "Time": arr["a"], "Type": "in"}))
    recs.append(pd.DataFrame({"Node": "Arrival", "Time": arr["b"], "Type": "out"}))

    add("Check-in", "Check-in Start", "Check-in End", df)
    for side in ["West", "East"]:
        add(f"Security {side}", "Security Queue Start", "Security Queue End",
            df[df["Checkpoint"] == side])
    for k in range(1, NUM_AIRSIDES + 1):
        add(f"Airside {k}", "Airside Arrival", "Boarding Time",
            df[df["Airside"] == k])

    gate = df[["Boarding Time", "Gate"]].dropna(subset=["Boarding Time"]).copy()
    gate["Node"] = "Gate " + gate["Gate"].astype(int).astype(str)
    recs.append(pd.DataFrame({"Node": gate["Node"], "Time": gate["Boarding Time"], "Type": "in"}))
    return pd.concat(recs, ignore_index=True)


if __name__ == "__main__":
    A = adjacency()
    print(f"{NUM_NODES} nodes; edges: {int(A.sum())}")
    print(NODES)
    print("normalized symmetric:", np.allclose(normalized_adjacency(), normalized_adjacency().T))
