"""Canonical airport graph — layout dispatcher.

This remains the SINGLE import point for node names, node order, and the
adjacency matrix (the node-ordering fix). The concrete layout is selected
with the AIRPORT_LAYOUT environment variable:

    AIRPORT_LAYOUT=linear   (default) 4 process nodes + 20 gates   -> airport_graph_linear
    AIRPORT_LAYOUT=mco      MCO-inspired: 2 checkpoints, APM,
                            4 airsides + 24 gates (32 nodes)       -> airport_graph_mco

Every script (build_features, train_models, baselines, plots) imports from
here, so switching layouts never requires touching them:

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
elif LAYOUT == "real":
    # real airport: checkpoint graph written by real_airport.py into REAL_LAYOUT_DIR
    import sys
    sys.path.insert(0, os.environ.get("REAL_LAYOUT_DIR", "."))
    from airport_graph_real import *         # noqa: F401,F403
    from airport_graph_real import (NODES, NODE_INDEX, NUM_NODES, CHECKPOINTS,
                                    GATES, NUM_GATES, SCHED_DEP_NODES, GATE_HUB,
                                    LAYOUT_NAME, build_graph, adjacency,
                                    normalized_adjacency, node_flow_events)
else:
    raise ValueError(f"Unknown AIRPORT_LAYOUT={LAYOUT!r}; use 'linear', 'mco' or 'real'")


if __name__ == "__main__":
    import numpy as np
    print(f"layout={LAYOUT_NAME}  nodes={NUM_NODES}  edges={int(adjacency().sum())}")
    print(NODES)
