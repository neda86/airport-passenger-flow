# Airport Passenger Flow Forecasting with GCN-LSTM

A digital-twin framework for forecasting passenger flow at airport checkpoints.
A **GCN-LSTM** couples graph convolutions over the airport's checkpoint/gate
graph (spatial structure) with LSTM temporal modeling to forecast per-node
passenger volumes over a **3-hour horizon**, supporting proactive resource
allocation and service optimization.

> Paper in preparation (target: TRB 2027).

## Overview

- **Airport graph:** a canonical 24-node directed graph — Arrival, Check-in,
  Security, Boarding, and 20 gates — defined once in `src/airport_graph.py`
  (single source of truth for node order and adjacency).
- **Synthetic data:** passenger trajectories are produced by a configurable
  simulator (`src/generate_data.py`, exploratory version in
  `notebooks/Airport_Passenger_generator.ipynb`); no operational data is used.
- **Model:** GCN-LSTM trained on node-level demand tensors
  (`src/train_models.py`; original Colab implementation preserved in
  `src/colab_gcn_lstm_original.py`).
- **Baselines:** time-series foundation models — Chronos and Chronos-2
  zero-shot (`src/baseline_chronos.py`, `src/baseline_chronos2.py`),
  **LoRA fine-tuned Chronos-2** (`src/finetune_chronos2.py`), and an ST-LLM
  baseline (`src/st_llm.py`).
- **Evaluation:** multi-seed runs and distribution-shift analysis
  (`src/eval_shift.py`); per-horizon metrics in `results/` (1-hour bins,
  3-step horizon).

## Repository layout

```
src/         graph definition, simulator, features, models, baselines, eval
notebooks/   passenger-flow generator (exploratory)
results/     forecast metrics (JSON; multiple seeds and baselines)
data/        NOT tracked - see data/README.md for the tensor/schema notes
```

## Setup

```bash
pip install -r requirements.txt
```

## Typical workflow

```bash
python src/generate_data.py      # simulate passenger flows on the airport graph
python src/build_features.py     # node-level feature/target tensors
python src/train_models.py       # GCN-LSTM training
python src/eval_shift.py         # evaluation incl. distribution shift
python src/make_plots.py         # figures
```

## Author

Neda Ghafouri — PhD researcher, University of Central Florida.
