"""Simulator sensitivity / parameter-shift evaluation.

Trains a small set of models ONCE on the baseline tensors and evaluates each
of them, without retraining, on tensors built from datasets generated with
different simulator parameters (load factor, show-up profile, lane capacity,
disruption frequency). Normalisation statistics come from the baseline
training split and are applied unchanged to the shifted data, and every
window of a shifted dataset is scored, so the numbers answer "does a model
trained under one parameter setting still work under another" (the
train-on-one / test-on-another analysis of the paper).

Models are given as NAME=spatial/temporal/sched, e.g.
    LSTM+Sched=none/lstm/1  TDN+Sched=none/tdn/1  Hybrid=none/tdn_noq_both/1  LSTM=none/lstm/0

Usage (from the pipeline directory, AIRPORT_LAYOUT=mco):
    python eval_sensitivity.py --data data/tensors_mco_1h_h3.npz \
        --shift lf_low=data_lf75/tensors.npz lf_high=data_lf92/tensors.npz \
                showup_early=data_su120/tensors.npz lanes_80=data_lane80/tensors.npz \
                delays_30=data_dis/tensors_mcodis_real_1h_h3.npz \
        --models "LSTM+Sched=none/lstm/1,TDN+Sched=none/tdn/1,Hybrid=none/tdn_noq_both/1" \
        --seed 42 --out results/sensitivity_seed42.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from airport_graph import adjacency, NODES, CHECKPOINTS  # noqa: E402
from train_models import (chrono_split, normalize_with_train_stats, loaders,  # noqa: E402
                          STModel, train, predict, metrics)

CP = np.array([n in CHECKPOINTS for n in NODES])


def scoped(y, p):
    p = np.maximum(p, 0)
    r = {"overall": metrics(y, p), "checkpoints": metrics(y, p, CP), "gates": metrics(y, p, ~CP)}
    r["macro_RMSE"] = (r["checkpoints"]["RMSE"] + r["gates"]["RMSE"]) / 2
    return r


def main(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    base = np.load(a.data, allow_pickle=True)
    X, Y, S = base["X"], base["Y"], base["S"]
    feats = list(base["features"])
    horizon, n_feat, sched_k = Y.shape[2], X.shape[2], S.shape[2]

    raw = chrono_split(X, S, Y)
    splits, (tmin, tmax) = normalize_with_train_stats(raw)
    train_dl, val_dl, test_dl = loaders(splits)
    denorm = lambda z: tmin + z * (tmax - tmin)
    # recover the feature statistics so shifted tensors get the SAME transform
    Xtr, Str = raw[0][0], raw[0][1]
    mu, sd = Xtr.mean(axis=(0, 1, 3), keepdims=True), Xtr.std(axis=(0, 1, 3), keepdims=True)
    sd[sd == 0] = 1.0
    smu, ssd = Str.mean(axis=(0, 1, 3), keepdims=True), Str.std(axis=(0, 1, 3), keepdims=True)
    ssd[ssd == 0] = 1.0
    scale = (tmax - tmin) or 1.0

    shifted = {}
    for item in a.shift:
        name, path = item.split("=", 1)
        d = np.load(path, allow_pickle=True)
        assert list(d["features"]) == feats, f"feature mismatch in {path}"
        Xs, Ys, Ss = d["X"], d["Y"], d["S"]
        # evaluation loader: fixed order, no dropped batch (loaders() treats index 0 as training)
        ds = TensorDataset(torch.tensor(((Xs - mu) / sd).astype(np.float32)),
                           torch.tensor(((Ss - smu) / ssd).astype(np.float32)),
                           torch.tensor(((Ys - tmin) / scale).astype(np.float32)))
        dl = DataLoader(ds, batch_size=32, shuffle=False, drop_last=False)
        shifted[name] = (Ys, dl)

    A = adjacency()
    results = {"_config": {"data": a.data, "seed": a.seed, "shifts": a.shift}}
    for spec in a.models.split(","):
        name, cfg = spec.split("=")
        sp, tp, use_s = cfg.split("/")
        k = sched_k if use_s == "1" else 0
        print(f"\n--- {name}: spatial={sp} temporal={tp} sched_k={k} ---")
        m = train(STModel(A, len(NODES), n_feat, horizon, sched_k=k, spatial=sp, temporal=tp),
                  train_dl, val_dl, a.epochs)
        yb, pb = map(denorm, predict(m, test_dl))
        results[name] = {"base_test": scoped(yb, pb)}
        for sname, (Ys, dl) in shifted.items():
            _, ps = predict(m, dl)
            results[name][sname] = scoped(Ys, denorm(ps))
        for split, r in results[name].items():
            print(f"  {split:14s} RMSE {r['overall']['RMSE']:7.2f}  process {r['checkpoints']['RMSE']:7.2f}"
                  f"  gates {r['gates']['RMSE']:6.2f}  macro {r['macro_RMSE']:6.2f}")

    out = a.out or a.data.replace(".npz", f"_sensitivity_seed{a.seed}.json")
    json.dump(results, open(out, "w"), indent=2)
    print("\nsaved ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--shift", nargs="+", required=True, help="name=path/to/tensors.npz ...")
    ap.add_argument("--models", default="LSTM+Sched=none/lstm/1,TDN+Sched=none/tdn/1,Hybrid=none/tdn_noq_both/1")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    main(ap.parse_args())
