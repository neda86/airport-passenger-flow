"""Terminal layout dispatcher: node names, node order and adjacency.

Every script imports the graph from here, so the node order of the tensors
and the row order of the adjacency matrix always agree. The layout is chosen
with the AIRPORT_LAYOUT environment variable:

    AIRPORT_LAYOUT=mco      MCO-inspired terminal: two checkpoints, APM,
                            four airside concourses, 24 gates (32 nodes)  -> airport_graph_mco
    AIRPORT_LAYOUT=linear   (default) linear terminal: 4 process nodes
                            and 20 gates (24 nodes)                       -> airport_graph_linear

    AIRPORT_LAYOUT=mco python build_features.py ...
    AIRPORT_LAYOUT=mco python train_models.py ...
"""
import os

LAYOUT = os.environ.get("AIRPORT_LAYOUT", "linear").strip().lower()

if LAYOUT == "mco":
    from airport_graph_mco import *          # noqa: F401,F403
    from airport_graph_mco import (NODES, NODE_INDEX, NUM_NODES, CHECKPOINTS,
                                   GATES, NUM_GATES, SCHED_DEP_NODES, GATE_HUB,
                                   LAYOUT_NAME, build_graph, adjacency,
                                   normalized_adjacency, node_flow_events)
elif LAYOUT == "linear":
    from airport_graph_linear import *       # noqa: F401,F403
    from airport_graph_linear import (NODES, NODE_INDEX, NUM_NODES, CHECKPOINTS,
                                      GATES, NUM_GATES, SCHED_DEP_NODES, GATE_HUB,
                                      LAYOUT_NAME, build_graph, adjacency,
                                      normalized_adjacency, node_flow_events)
else:
    raise ValueError(f"Unknown AIRPORT_LAYOUT={LAYOUT!r}; use 'mco' or 'linear'")


if __name__ == "__main__":
    import numpy as np
    print(f"layout={LAYOUT_NAME}  nodes={NUM_NODES}  edges={int(adjacency().sum())}")
    print(NODES)
