# Airport Passenger Flow Forecasting with GAT-LSTM

A digital-twin framework for predicting passenger flow at airport security
checkpoints. The model combines **Graph Attention Networks** (spatial
dependencies across airport nodes) with **LSTM** temporal modeling to forecast
checkpoint passenger volumes over a **3-hour horizon**, supporting proactive
resource allocation and service optimization.

> Paper in preparation (target: TRB 2027). Code is being released incrementally
> as the paper is finalized.

## Approach

- **Graph construction:** airport checkpoints/zones as nodes; passenger-transfer
  and adjacency relations as edges.
- **Spatial encoder:** Graph Attention Network (GAT) layers learn which
  neighboring nodes matter for each checkpoint at each time step.
- **Temporal head:** LSTM over the GAT-encoded sequence produces multi-step
  forecasts (3-hour horizon).
- **Digital-twin loop:** forecasts feed a checkpoint-resource simulation for
  staffing/lane decisions.

## Repository layout

```
src/        model, training, and evaluation code
configs/    experiment configurations
notebooks/  exploratory analysis
data/       NOT tracked - see data/README.md
```

## Data

Passenger-flow data is **not distributed** with this repository (operational
sensitivity). See `data/README.md` for the expected input schema so the
pipeline can be run on your own data.

## Setup

```bash
pip install -r requirements.txt
```

## Status

- [x] GAT-LSTM model and training loop
- [x] 3-hour horizon evaluation
- [ ] Paper (TRB 2027, in preparation)
- [ ] Full documentation

## Author

Neda Ghafouri — PhD researcher, University of Central Florida.
