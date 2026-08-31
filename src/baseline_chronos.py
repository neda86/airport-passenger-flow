"""Zero-shot Chronos-Bolt baseline on the airport flow tensors.

Time-series foundation models (Chronos, TimesFM, Moirai) are the expected
modern baselines in 2026. This runs amazon/chronos-bolt-small zero-shot,
per node, at every test forecast origin, using a long context window
(not just the 12-step model input — TSFMs benefit from longer history).

Univariate per node: no graph, no schedule — measures how far pretrained
temporal knowledge alone gets you on this task.

Usage: baseline_chronos.py --data tensors_1h_h3.npz [--context 512]
"""
import argparse
import json
import numpy as np
import torch

from airport_graph import NODES, CHECKPOINTS
from train_models import metrics


def reconstruct_series(X, Y):
    """Rebuild the full per-node Flow_In series from overlapping windows.
    X windows step by 1, so series[t] = X[t, :, 0, 0] plus tail."""
    in_seq = X.shape[3]
    head = X[:, :, 0, 0]                     # (samples, nodes) — first lag of each window
    tail = X[-1, :, 0, 1:].T                 # (in_seq-1, nodes) — rest of last window
    series = np.concatenate([head, tail], axis=0)  # (T_hist, nodes)
    return series, in_seq


def main(data_path, context_len, batch_origins=64):
    d = np.load(data_path, allow_pickle=True)
    X, Y = d["X"], d["Y"]
    horizon = Y.shape[2]
    n = X.shape[0]
    test_start = int(n * 0.85)
    series, in_seq = reconstruct_series(X, Y)   # series[s + in_seq - 1] = last obs before origin s

    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32)

    origins = range(test_start, n)
    preds = np.zeros((n - test_start, len(NODES), horizon), dtype=np.float32)
    batch = []
    slots = []
    for k, s in enumerate(origins):
        end = s + in_seq          # first unobserved index in series
        ctx = series[max(0, end - context_len):end]   # (ctx_len, nodes)
        for node in range(len(NODES)):
            batch.append(torch.tensor(ctx[:, node]))
            slots.append((k, node))
        if len(batch) >= batch_origins * len(NODES) or s == n - 1:
            q, mean = pipe.predict_quantiles(batch, prediction_length=horizon,
                                             quantile_levels=[0.5])
            for (kk, node), m in zip(slots, mean):
                preds[kk, node] = m.numpy()
            batch, slots = [], []
        if (k + 1) % 64 == 0:
            print(f"  {k + 1}/{n - test_start} origins")

    y_true = Y[test_start:]
    preds = np.maximum(preds, 0)
    cp_mask = np.array([nm in CHECKPOINTS for nm in NODES])
    res = {"overall": metrics(y_true, preds),
           "checkpoints": metrics(y_true, preds, cp_mask),
           "gates": metrics(y_true, preds, ~cp_mask)}
    print("\n== Chronos-Bolt (zero-shot) ==")
    for scope, m in res.items():
        print(f"  {scope:12s} RMSE {m['RMSE']:8.2f}  MAE {m['MAE']:7.2f}  "
              f"R2 {m['R2']:6.3f}  MAPE(nz) {m['MAPE_nonzero']:7.1f}%")

    out = data_path.replace(".npz", "_chronos.json")
    json.dump(res, open(out, "w"), indent=2)
    np.savez_compressed(data_path.replace(".npz", "_chronos_preds.npz"),
                        y_true=y_true, pred_Chronos=preds)
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--context", type=int, default=512)
    args = ap.parse_args()
    main(args.data, args.context)
