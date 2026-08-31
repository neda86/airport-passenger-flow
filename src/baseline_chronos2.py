"""Chronos-2 covariate-informed zero-shot baseline.

Chronos-2 (arXiv:2510.15821) supports known-future covariates natively, so —
unlike the univariate Chronos-Bolt baseline — it receives EXACTLY the same
schedule information as the +Sched neural models: sched_board and sched_dep,
past values as past_covariates and the forecast-horizon values as
future_covariates. Same test split, same metrics. This answers the
"you handicapped the foundation model" objection directly.

Usage: baseline_chronos2.py --data tensors_1h_h3.npz [--context 512]
"""
import argparse
import json
import numpy as np
import torch

from airport_graph import NODES, CHECKPOINTS
from train_models import metrics

SCHED_NAMES = ["sched_board", "sched_dep"]


def reconstruct_series(X, feat_idx):
    """Full per-node series for feature feat_idx from overlapping windows
    (windows step by 1). Returns (T_hist, nodes)."""
    head = X[:, :, feat_idx, 0]
    tail = X[-1, :, feat_idx, 1:].T
    return np.concatenate([head, tail], axis=0)


def main(data_path, context_len, batch_origins=16):
    d = np.load(data_path, allow_pickle=True)
    X, Y, S = d["X"], d["Y"], d["S"]
    feats = list(d["features"])
    horizon = Y.shape[2]
    n = X.shape[0]
    in_seq = X.shape[3]
    test_start = int(n * 0.85)

    tgt_series = reconstruct_series(X, 0)                     # Flow_In
    sched_idx = [feats.index(nm) for nm in SCHED_NAMES]
    sched_series = {nm: reconstruct_series(X, i)
                    for nm, i in zip(SCHED_NAMES, sched_idx)}

    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2", device_map="cpu", torch_dtype=torch.float32)

    preds = np.zeros((n - test_start, len(NODES), horizon), dtype=np.float32)
    batch, slots = [], []

    def flush():
        nonlocal batch, slots
        if not batch:
            return
        out = pipe.predict_quantiles(batch, prediction_length=horizon,
                                     quantile_levels=[0.5])
        # predict_quantiles returns (quantiles, mean)
        mean = out[1] if isinstance(out, tuple) else out
        for (kk, node), m in zip(slots, mean):
            preds[kk, node] = np.asarray(m).reshape(-1)[:horizon]
        batch, slots = [], []

    for k, s in enumerate(range(test_start, n)):
        end = s + in_seq
        lo = max(0, end - context_len)
        for node in range(len(NODES)):
            past_cov = {nm: sched_series[nm][lo:end, node]
                        for nm in SCHED_NAMES}
            fut_cov = {nm: S[s, node, j, :]
                       for j, nm in enumerate(SCHED_NAMES)}
            batch.append({"target": tgt_series[lo:end, node],
                          "past_covariates": past_cov,
                          "future_covariates": fut_cov})
            slots.append((k, node))
        if len(batch) >= batch_origins * len(NODES):
            flush()
        if (k + 1) % 32 == 0:
            print(f"  {k + 1}/{n - test_start} origins")
    flush()

    y_true = Y[test_start:]
    preds = np.maximum(preds, 0)
    cp_mask = np.array([nm in CHECKPOINTS for nm in NODES])
    res = {"overall": metrics(y_true, preds),
           "checkpoints": metrics(y_true, preds, cp_mask),
           "gates": metrics(y_true, preds, ~cp_mask)}
    print("\n== Chronos-2 (zero-shot, schedule covariates) ==")
    for scope, m in res.items():
        print(f"  {scope:12s} RMSE {m['RMSE']:8.2f}  MAE {m['MAE']:7.2f}  "
              f"R2 {m['R2']:6.3f}  MAPE(nz) {m['MAPE_nonzero']:7.1f}%")

    out = data_path.replace(".npz", "_chronos2.json")
    json.dump(res, open(out, "w"), indent=2)
    np.savez_compressed(data_path.replace(".npz", "_chronos2_preds.npz"),
                        y_true=y_true, pred_Chronos2=preds)
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--context", type=int, default=512)
    args = ap.parse_args()
    main(args.data, args.context)
