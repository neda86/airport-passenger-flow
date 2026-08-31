# src/ module guide

| file | role |
|---|---|
| `airport_graph.py` | Canonical airport graph: node names, node order, adjacency. Single source of truth imported by everything else. |
| `generate_data.py` | Synthetic passenger-flow simulator over the airport graph. |
| `build_features.py` | Node-level feature/target tensor construction. |
| `train_models.py` | GCN-LSTM training. |
| `colab_gcn_lstm_original.py` | Original Colab implementation (preserved for provenance; paths reference Colab/Drive). |
| `baseline_chronos.py` / `baseline_chronos2.py` | Chronos / Chronos-2 zero-shot TSFM baselines. |
| `finetune_chronos2.py` | LoRA fine-tuning of Chronos-2. |
| `colab_tsfm_baselines.py` | Colab runner for TSFM baselines. |
| `st_llm.py` | ST-LLM baseline. |
| `eval_shift.py` | Evaluation, including distribution-shift analysis. |
| `make_plots.py` | Figures. |
