"""Distribution-shift (generalization) evaluation.

Trains models on the BASELINE simulation config, then evaluates them —
without any retraining — on a DIFFERENTLY-CONFIGURED simulation run
(e.g. denser flight schedule, fewer security lanes, different seed).
This tests whether the models learned transferable structure or merely
memorized the generator's regularities, and answers the "synthetic data
flatters high-capacity models" objection with an experiment.

All normalization statistics come from the BASE train split only and are
applied unchanged to the shifted data (exactly what deployment under
drift looks like). Every window of the shifted dataset is test data.

Typical shifted config (30% denser traffic, one security lane closed,
different random seed => different flights, gates, passengers):

  python generate_data.py --out ../data_shift --seed 7 \
      --flight-scale 1.3 --security-lanes 6
  python build_features.py --passengers ../data_shift/passengers.csv \
      --flights ../data_shift/flights.csv --freq 1h --in-seq 12 --tar-seq 3 \
      --out ../data_shift/tensors_1h_h3_shift.npz
  python eval_shift.py --data ../data/tensors_1h_h3.npz \
      --shift-data ../data_shift/tensors_1h_h3_shift.npz
"""
import argparse
import json
import numpy as np
import torch

from airport_graph import adjacency, NODES, CHECKPOINTS
from train_models import (chrono_split, loaders, STModel, train, predict,
                          metrics, DEVICE)

MODEL_SPECS = {  # name -> (spatial, sched?)
    "GCN-LSTM+Sched": ("gcn", True),
    "LSTM+Sched": ("none", True),
    "GCN-LSTM": ("gcn", False),
    "LSTM": ("none", False),
}


def fit_stats(Xtr, Str, Ytr):
    mu = Xtr.mean(axis=(0, 1, 3), keepdims=True)
    sd = Xtr.std(axis=(0, 1, 3), keepdims=True)
    sd[sd == 0] = 1.0
    smu = Str.mean(axis=(0, 1, 3), keepdims=True)
    ssd = Str.std(axis=(0, 1, 3), keepdims=True)
    ssd[ssd == 0] = 1.0
    tmin, tmax = float(Ytr.min()), float(Ytr.max())
    return mu, sd, smu, ssd, tmin, (tmax - tmin) or 1.0


def apply_stats(stats, X, S, Y):
    mu, sd, smu, ssd, tmin, scale = stats
    return (X - mu) / sd, (S - smu) / ssd, (Y - tmin) / scale


def scoped(y, p):
    cp = np.array([nm in CHECKPOINTS for nm in NODES])
    return {"overall": metrics(y, p), "checkpoints": metrics(y, p, cp),
            "gates": metrics(y, p, ~cp)}


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base = np.load(args.data, allow_pickle=True)
    Xb, Yb, Sb = base["X"], base["Y"], base["S"]
    shift = np.load(args.shift_data, allow_pickle=True)
    Xs, Ys, Ss = shift["X"], shift["Y"], shift["S"]
    assert list(base["features"]) == list(shift["features"]), "feature mismatch"
    horizon, n_feat, sched_k = Yb.shape[2], Xb.shape[2], Sb.shape[2]

    (XtrR, StrR, YtrR), va_raw, te_raw = chrono_split(Xb, Sb, Yb)
    stats = fit_stats(XtrR, StrR, YtrR)
    tmin, scale = stats[4], stats[5]
    denorm = lambda a: tmin + a * scale

    tr = apply_stats(stats, XtrR, StrR, YtrR)
    va = apply_stats(stats, *va_raw)
    te = apply_stats(stats, *te_raw)
    sh = apply_stats(stats, Xs, Ss, Ys)          # base-train stats, unchanged
    train_dl, val_dl, test_dl = loaders([tr, va, te])
    (shift_dl,) = loaders([sh])[:1]

    A = adjacency()
    results = {}
    names = [s.strip() for s in args.models.split(",")]
    for name in names:
        sp, use_sched = MODEL_SPECS[name]
        k = sched_k if use_sched else 0
        print(f"\n--- {name} (train on base config) ---")
        m = train(STModel(A, len(NODES), n_feat, horizon, sched_k=k,
                          spatial=sp, temporal="lstm"),
                  train_dl, val_dl, args.epochs)
        yb, pb = map(denorm, predict(m, test_dl))
        ys, ps = map(denorm, predict(m, shift_dl))
        results[name] = {"base_test": scoped(yb, np.maximum(pb, 0)),
                         "shifted": scoped(ys, np.maximum(ps, 0))}
        for split in ("base_test", "shifted"):
            r = results[name][split]
            print(f"  {split:9s} RMSE {r['overall']['RMSE']:7.2f} "
                  f"R2 {r['overall']['R2']:6.3f} | gates RMSE "
                  f"{r['gates']['RMSE']:6.2f} R2 {r['gates']['R2']:6.3f}")

    # Moving average reference on the shifted set (training-free calibration).
    p_ma = np.repeat(Xs[:, :, 0, :].mean(axis=2, keepdims=True), horizon, axis=2)
    results["MovingAverage"] = {"shifted": scoped(Ys, p_ma)}

    out = args.shift_data.replace(".npz", f"_shift_eval_seed{args.seed}.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="baseline-config tensors")
    ap.add_argument("--shift-data", required=True, help="shifted-config tensors")
    ap.add_argument("--models", default="GCN-LSTM+Sched,LSTM+Sched,GCN-LSTM,LSTM")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
