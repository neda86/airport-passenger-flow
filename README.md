# Airport terminal passenger-flow forecasting with known-future flight information

Code for the paper *The Schedule Is the Signal: Forecasting Airport Terminal Passenger Flows with Known-Future Flight Information* (N. Ghafouri and K. Thakkar, University of Central Florida, 2026, under review).

The question behind the paper is simple: how much of the terminal forecasting problem is already answered by the published flight schedule, and what does a learned model add on top of it? To answer it we built a discrete-event simulator of a hub terminal inspired by Orlando International Airport (two parallel security checkpoints, an automated people mover to four airside concourses, 24 gates), turned the schedule into two known-future covariates, and fed the same information to a family of graph and temporal models and to the Chronos-2 foundation model. This repository has everything needed to regenerate the data, train the models and rerun the controls described in the paper. It does not ship results; the numbers in the paper come out of the commands below.

## Layout

```
src/        simulator, feature construction, models, baselines, analysis, figures
scripts/    reproduce.sh, the experiment sequence of the paper in one file
data/       generated data goes here (not tracked, see data/README.md)
notebooks/  the early exploratory passenger generator
```

Everything in `src/` is a plain Python script with a command-line interface; there is no package to install. Run the scripts from inside `src/` or put `src/` on `PYTHONPATH`.

## Setup

Python 3.10 or newer.

```
pip install -r requirements.txt
```

`torch`, `numpy`, `pandas`, `networkx` and `matplotlib` are enough for the simulator and the supervised models. The Chronos baselines additionally need `chronos-forecasting>=2.0` (and `peft` for LoRA fine-tuning), which pull in `transformers`. A GPU is not required for the supervised models (a training run on the hourly tensors takes a few minutes on a T4 and a good deal longer on a laptop CPU); fine-tuning Chronos-2 is not practical without one.

One environment variable selects the terminal layout for every script:

```
export AIRPORT_LAYOUT=mco      # the 32-node MCO-inspired terminal used in the paper
export AIRPORT_LAYOUT=linear   # the earlier 24-node linear terminal (transfer experiment)
```

`src/airport_graph.py` reads it and exposes the node list, node order and adjacency to all other scripts, so the node ordering of the tensors and of the graph operators always agree.

## Step by step

### 1. Generate the terminal data

```
cd src
AIRPORT_LAYOUT=mco python generate_mco.py --out ../data --days 90 --seed 42
```

This writes `flights.csv` (one row per flight: scheduled and realized times, gate, seats, load factor, passengers) and `passengers.csv` (one row per passenger with the time of every event from arrival at the terminal to boarding). The defaults are the paper's baseline: eight daily banks, four seat classes, Beta-distributed load factors, log-normal show-up times, finite-capacity kiosk, desk and lane queues. The other datasets of the paper are variations of the same call:

```
# 30% delayed flights and 5% gate changes (Section 5.4)
python generate_mco.py --out ../data_dis --days 90 --delay-frac 0.3 --gate-change-frac 0.05

# sensitivity datasets (Section 5.8): one parameter changed at a time
python generate_mco.py --out ../data_lf75  --days 90 --lf-mean 0.75
python generate_mco.py --out ../data_lf92  --days 90 --lf-mean 0.92
python generate_mco.py --out ../data_su125 --days 90 --showup-scale 1.25
python generate_mco.py --out ../data_su80  --days 90 --showup-scale 0.80
python generate_mco.py --out ../data_lane75 --days 90 --lane-scale 0.75
python generate_mco.py --out ../data_seed7 --days 90 --seed 7

# the linear terminal of the transfer experiment
AIRPORT_LAYOUT=linear python generate_data.py --out ../data_lin
```

`simulator_validation.py --data ../data` prints the aggregate statistics we compare with the public airport figures in the paper (departures and passengers per day, seats per departure, load factor, hourly and weekday seat shares, checkpoint throughput, show-up profile).

### 2. Build the tensors

`build_features.py` aggregates the passenger events on a regular time grid, computes the two schedule covariates from `flights.csv` and writes a compressed `.npz` with the history tensor `X`, the target `Y` and the known-future covariate tensor `S`.

```
# hourly bins, 12 hours in, 3 hours out (the main setting)
python build_features.py --passengers ../data/passengers.csv --flights ../data/flights.csv \
    --freq 1h --in-seq 12 --tar-seq 3 --out ../data/tensors_mco_1h_h3.npz

# 15-minute bins, 12 steps in, 12 steps (3 h) out
python build_features.py --passengers ../data/passengers.csv --flights ../data/flights.csv \
    --freq 15min --in-seq 12 --tar-seq 12 --out ../data/tensors_mco_15min.npz

# horizon sweep: 24-hour input, H in {1, 3, 6, 12, 24}
python build_features.py ... --freq 1h --in-seq 24 --tar-seq 6 --out ../data/tensors_mco_1h_in24_H6.npz
```

Two options decide what the covariates may know, which is the information question of the paper:

* `--pax realized|expected`. `realized` uses the passenger count the simulator drew for each flight; `expected` uses seats times the mean load factor, which is what an operator has before departure. The main table uses realized loads, Section 5.5 repeats the headline models with expected loads.
* `--schedule realized|published|mixed`. On the disrupted data, `published` builds the covariates from the schedule the operator sees in advance, `realized` from what actually happened (the oracle rows), and `mixed` uses the published schedule for the landside covariate and the realized gate and boarding time for the gate covariate.

### 3. Train the supervised models

`train_models.py` trains every combination of a spatial operator and a temporal encoder on one tensor file, with and without the schedule covariates, and writes the metrics and the test predictions next to it.

```
python train_models.py --data ../data/tensors_mco_1h_h3.npz --spatials none,gcn,gat --temporals lstm,transformer,tdn --seed 42
```

Output: `tensors_mco_1h_h3_results.json` (metrics per model, overall, process nodes and gates) and `tensors_mco_1h_h3_preds.npz` (test predictions of every model, used by the staffing and bootstrap scripts). With `--seed 7` or `--seed 123` the file names get a `_seed7` / `_seed123` suffix; `aggregate_seeds.py` and `analysis_macro.py` pool them.

Spatial operators: `none` (a per-node linear map; the node embeddings are still concatenated before the encoder, so this is the "dense learned mixing" of the paper), `gcn`, `gat`, `diffusion` (DCRNN-style, uses edge direction), `cheb` (Chebyshev, order 3), `isolated` (each node encoded on its own) and `gcn_shuffled` (a permuted adjacency). Temporal encoders: `lstm`, `transformer`, `tdn`, plus the TDN ablations `tdn_noq`, `tdn_nopsi`, `tdn_noctx`, `tdn_noprior`, `tdn_head`, `tdn_both`, and the encoder-isolation variants `lstm_ctx` and `lstm_ctxboth`. `schedonly` is the control that sees the covariates and no flow history.

The names in the paper map to these flags as follows.

| paper | spatial / temporal | name in the results file |
|---|---|---|
| LSTM, LSTM+Sched | none / lstm | `LSTM`, `LSTM+Sched` |
| GCN-LSTM+Sched, GAT-TDN+Sched, ... | gcn or gat / lstm or tdn | `GCN-LSTM+Sched`, `GAT-TDN+Sched` |
| TDN+Sched | none / tdn | `TDN+Sched` |
| TDN hybrid (token, prior, head) | none / tdn_both | `TDN_BOTH+Sched` |
| TDN hybrid, last-step readout | none / tdn_noq_both | `TDN_NOQ_BOTH+Sched` |
| LSTM with TDN context and prior | none / lstm_ctx | `LSTM_CTX+Sched` |
| LSTM with TDN context, prior, and head | none / lstm_ctxboth | `LSTM_CTXBOTH+Sched` |
| SchedOnly | none / schedonly | `SchedOnly` |
| isolated nodes, shuffled GCN | isolated or gcn_shuffled / lstm or tdn | `Iso-...`, `ShufGCN-...` |
| directed diffusion, Chebyshev | diffusion or cheb / lstm or tdn | `Diff-...`, `Cheb-...` |

Training details (Adam, learning rate 1e-3, weight decay 1e-5, batch 32, early stopping on the validation loss with patience 15, chronological 70/15/15 split, normalisation fitted on the training split) are fixed in the script and are the ones reported in the paper's appendix.

### 4. Foundation-model baselines

```
python baseline_chronos.py  --data ../data/tensors_mco_1h_h3.npz            # Chronos-Bolt, univariate zero-shot
python baseline_chronos2.py --data ../data/tensors_mco_1h_h3.npz            # Chronos-2 zero-shot with the schedule covariates
python baseline_chronos2.py --data ../data/tensors_mco_1h_h3.npz --no-cov   # Chronos-2 zero-shot without them
python finetune_chronos2.py --data ../data/tensors_mco_1h_h3.npz            # Chronos-2 LoRA fine-tuning with covariates
python finetune_chronos2.py --data ../data/tensors_mco_1h_h3.npz --no-covariates
```

The four Chronos-2 runs form the two-by-two design of the paper (with or without covariates, with or without fine-tuning). Fine-tuning uses the library's LoRA mode, 1,000 steps, learning rate 1e-5, batch 32, 512-step context, and keeps the checkpoint with the lowest validation loss.

### 5. Controls

Which information the model is allowed to see is decided when the tensors are built (step 2), not in the model. The disruption experiment (Section 5.4) trains the same models on `data_dis` tensors built with `--schedule published --pax expected` (operator view), `--schedule realized` (oracle) and `--schedule mixed`. The leakage controls (Section 5.5) use `--pax expected` on the clean data and add `schedonly` to `--temporals`. The transfer experiment repeats the hourly setting on the linear terminal with `AIRPORT_LAYOUT=linear`. The exact sequence, including seeds, is in `scripts/reproduce.sh`.

### 6. Sensitivity to the simulator parameters

`eval_sensitivity.py` trains a model once on the baseline tensors and evaluates the same trained model, without retraining, on tensors built from the modified datasets of step 1. Normalisation statistics stay those of the baseline training split.

```
python eval_sensitivity.py --data ../data/tensors_mco_1h_h3.npz \
    --shift lf_0.75=../data_lf75/tensors_mco_1h_h3_lf75.npz lf_0.92=../data_lf92/tensors_mco_1h_h3_lf92.npz \
            showup_x1.25=../data_su125/tensors_mco_1h_h3_su125.npz showup_x0.80=../data_su80/tensors_mco_1h_h3_su80.npz \
            lanes_x0.75=../data_lane75/tensors_mco_1h_h3_lane75.npz delays_30pct=../data_dis/tensors_mcodis_real_1h_h3.npz \
            new_seed=../data_seed7/tensors_mco_1h_h3_seed7.npz \
    --models "LSTM+Sched=none/lstm/1,TDN+Sched=none/tdn/1,Hybrid=none/tdn_noq_both/1,LSTM=none/lstm/0" \
    --seed 42 --out ../results/sensitivity_tensors_mco_1h_h3_seed42.json
python sensitivity_summary.py ../results     # mean and sd over the seeds found, plus LaTeX rows
```

### 7. From forecasts to a staffing decision, and uncertainty

```
python staffing_eval.py ../data/tensors_mco_1h_h3_preds.npz ../data/tensors_mco_1h_h3_chronos2_lora_preds.npz
python bootstrap_ci.py   ../data/tensors_mco_1h_h3_preds.npz --ref "LSTM+Sched" --B 2000
python analysis_macro.py ../results
```

`staffing_eval.py` turns each checkpoint forecast into an equivalent standard-lane requirement (passengers per hour divided by 112.5, the throughput of one simulated lane) and scores understaffed hours, surplus lane-hours, peak recall and the lane error. `bootstrap_ci.py` resamples the test days to give 95% intervals for the RMSE of each model and for paired differences against a reference model. `analysis_macro.py` reports the process-node, gate and macro RMSE (the unweighted mean of the two) for every results file, pooling seeds. `model_card.py` prints the parameter count of every model.

### 8. Figures

```
python make_layout_figure.py                      # terminal graph and dataset summary (writes to ../figures)
python make_framework_fig.py --out figures         # model family and TDN encoder
python make_plots_mco.py --preds ../data/tensors_mco_1h_h3_preds.npz --out figures   # time series, per-node errors
python make_horizon_fig.py --numbers horizon_numbers.json --out figures
```

`make_horizon_fig.py` reads a small JSON of the form `{model: {horizon: [overall, process, gates]}}` that you assemble from the horizon-sweep results files.

## Reproducing the paper

`scripts/reproduce.sh` runs steps 1 to 7 in the order we used, for the three seeds (42, 7, 123) where the paper reports a mean and standard deviation. It is idempotent: every step is skipped if its results file already exists, so it can be interrupted and restarted. The full sequence is a few GPU-hours on a T4; the Chronos-2 fine-tuning steps are the slow part and are skipped automatically when no GPU is found. All results land in `results/`, which is not tracked by git.

## Citation

```
@article{ghafouri2026schedule,
  author  = {Ghafouri, Neda and Thakkar, Kalp},
  title   = {The Schedule Is the Signal: Forecasting Airport Terminal Passenger Flows with Known-Future Flight Information},
  year    = {2026},
  note    = {Under review}
}
```

Questions about the code: Neda Ghafouri, neda.ghafouri@ucf.edu.
