# src/

All scripts read the terminal layout from the `AIRPORT_LAYOUT` environment variable (`mco` or `linear`) through `airport_graph.py`, which is the only place the node order and the adjacency are defined.

| script | what it does |
|---|---|
| `airport_graph.py`, `airport_graph_mco.py`, `airport_graph_linear.py` | node list, node order and adjacency of the two terminal layouts |
| `generate_mco.py` | discrete-event simulator of the MCO-inspired terminal; writes `flights.csv` and `passengers.csv` |
| `generate_data.py` | the earlier simulator of the linear terminal |
| `build_features.py` | bins the passenger events, computes the two schedule covariates and writes the `.npz` tensors |
| `train_models.py` | trains the spatial x temporal model family (with and without the schedule) and saves metrics and test predictions |
| `baseline_chronos.py`, `baseline_chronos2.py` | Chronos-Bolt and Chronos-2 zero-shot baselines, with or without the covariates |
| `finetune_chronos2.py` | Chronos-2 LoRA fine-tuning |
| `eval_sensitivity.py` | trains once on the baseline tensors and evaluates the same models on the modified datasets |
| `eval_shift.py` | earlier train-on-one / test-on-another evaluation for two tensor files |
| `staffing_eval.py` | turns checkpoint forecasts into equivalent screening-lane requirements and scores them |
| `bootstrap_ci.py` | bootstrap intervals over test days, including paired differences against a reference model |
| `analysis_macro.py`, `summarize_results.py`, `aggregate_seeds.py`, `sensitivity_summary.py` | pool the results files over seeds and print tables |
| `model_card.py` | parameter counts and the training configuration |
| `simulator_validation.py` | aggregate statistics of a generated dataset for comparison with public airport figures |
| `make_layout_figure.py`, `make_framework_fig.py`, `make_plots_mco.py`, `make_horizon_fig.py` | figures |
