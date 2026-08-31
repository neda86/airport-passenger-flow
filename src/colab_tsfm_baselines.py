"""Covariate-capable TSFM baselines that need a Colab GPU / modern torch.

These could not run on the local Intel Mac (torch capped at 2.2.2, jax/AVX
conflicts), so run this on Colab (T4 is fine). Chronos-2 already ran locally
(baseline_chronos2.py); this script adds the remaining covariate-capable
foundation models so the TSFM tier isn't a single data point:

  timesfm  — TimesFM 2.5 (Google), covariates via its XReg regression layer
  moirai   — Moirai-1.1-R (Salesforce), any-variate: covariates are packed
             as extra variates with known future values

Setup in a Colab cell (pick the model you're running):
  !pip install timesfm[torch]            # TimesFM
  !pip install uni2ts                    # Moirai
Upload: this file, airport_graph.py, train_models.py, and the tensors npz
(with S; built by build_features.py --flights ...).

Usage:
  python colab_tsfm_baselines.py --model timesfm --data tensors_1h_h3.npz
  python colab_tsfm_baselines.py --model moirai  --data tensors_1h_h3.npz

Both use the SAME protocol as every other baseline: chronological test split
(last 15%), per-node evaluation, metrics in passenger units via
train_models.metrics. Results -> <data>_<model>.json
"""
import argparse
import json
import numpy as np

from airport_graph import NODES, CHECKPOINTS
from train_models import metrics

SCHED_NAMES = ["sched_board", "sched_dep"]
CONTEXT = 512


def reconstruct_series(X, feat_idx):
    head = X[:, :, feat_idx, 0]
    tail = X[-1, :, feat_idx, 1:].T
    return np.concatenate([head, tail], axis=0)      # (T_hist, nodes)


def load(data_path):
    d = np.load(data_path, allow_pickle=True)
    X, Y, S = d["X"], d["Y"], d["S"]
    feats = list(d["features"])
    tgt = reconstruct_series(X, 0)
    sched = {nm: reconstruct_series(X, feats.index(nm)) for nm in SCHED_NAMES}
    return X, Y, S, tgt, sched


def run_timesfm(X, Y, S, tgt, sched):
    import timesfm
    horizon = Y.shape[2]
    n, in_seq = X.shape[0], X.shape[3]
    test_start = int(n * 0.85)
    if hasattr(timesfm, "TimesFm"):            # timesfm 1.x API
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend="gpu", horizon_len=horizon,
                                           context_len=CONTEXT),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"))
    else:                                      # timesfm 2.5 API
        cls = timesfm.TimesFM_2p5_200M_torch
        repo = "google/timesfm-2.5-200m-pytorch"
        if hasattr(cls, "from_pretrained"):
            tfm = cls.from_pretrained(repo)
        else:
            tfm = cls()
            try:
                tfm.load_checkpoint()          # some builds: no-arg download
            except TypeError:                  # this build: needs a local path
                from huggingface_hub import snapshot_download
                tfm.load_checkpoint(snapshot_download(repo))
        tfm.compile(timesfm.ForecastConfig(
            max_context=CONTEXT, max_horizon=max(horizon, 64),
            normalize_inputs=True, use_continuous_quantile_head=True,
            fix_quantile_crossing=True,
            return_backcast=True))   # required for forecast_with_covariates (XReg)
    preds = np.zeros((n - test_start, len(NODES), horizon), dtype=np.float32)
    for k, s in enumerate(range(test_start, n)):
        end = s + in_seq
        lo = max(0, end - CONTEXT)
        inputs = [tgt[lo:end, node] for node in range(len(NODES))]
        dyn_num = {nm: [np.concatenate([sched[nm][lo:end, node],
                                        S[s, node, j, :]])
                        for node in range(len(NODES))]
                   for j, nm in enumerate(SCHED_NAMES)}
        try:                                   # 1.x signature (needs freq)
            out = tfm.forecast_with_covariates(
                inputs=inputs, dynamic_numerical_covariates=dyn_num,
                freq=[0] * len(NODES))
        except TypeError:                      # 2.5 signature (no freq arg)
            out = tfm.forecast_with_covariates(
                inputs=inputs, dynamic_numerical_covariates=dyn_num)
        out = out[0] if isinstance(out, tuple) else out
        preds[k] = np.stack([np.asarray(o).reshape(-1)[:horizon] for o in out])
        if (k + 1) % 32 == 0:
            print(f"  {k + 1}/{n - test_start}")
    return preds


def run_moirai(X, Y, S, tgt, sched):
    import torch
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    horizon = Y.shape[2]
    n, in_seq = X.shape[0], X.shape[3]
    test_start = int(n * 0.85)
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-base"),
        prediction_length=horizon, context_length=CONTEXT,
        patch_size="auto", num_samples=32,
        target_dim=1, feat_dynamic_real_dim=len(SCHED_NAMES),
        past_feat_dynamic_real_dim=0)
    predictor = model.create_predictor(batch_size=64)
    from gluonts.dataset.common import ListDataset
    preds = np.zeros((n - test_start, len(NODES), horizon), dtype=np.float32)
    for k, s in enumerate(range(test_start, n)):
        end = s + in_seq
        lo = max(0, end - CONTEXT)
        entries = []
        for node in range(len(NODES)):
            fdr = np.stack([np.concatenate([sched[nm][lo:end, node],
                                            S[s, node, j, :]])
                            for j, nm in enumerate(SCHED_NAMES)])
            entries.append({"start": "2024-01-01 00:00",
                            "target": tgt[lo:end, node],
                            "feat_dynamic_real": fdr})
        ds = ListDataset(entries, freq="H")
        for node, f in enumerate(predictor.predict(ds)):
            preds[k, node] = f.mean[:horizon]
        if (k + 1) % 32 == 0:
            print(f"  {k + 1}/{n - test_start}")
    return preds


def main(model, data_path):
    X, Y, S, tgt, sched = load(data_path)
    preds = {"timesfm": run_timesfm, "moirai": run_moirai}[model](X, Y, S, tgt, sched)
    y_true = Y[int(X.shape[0] * 0.85):]
    preds = np.maximum(preds, 0)
    cp = np.array([nm in CHECKPOINTS for nm in NODES])
    res = {"overall": metrics(y_true, preds),
           "checkpoints": metrics(y_true, preds, cp),
           "gates": metrics(y_true, preds, ~cp)}
    print(f"\n== {model} (zero-shot, schedule covariates) ==")
    for scope, m in res.items():
        print(f"  {scope:12s} RMSE {m['RMSE']:8.2f}  MAE {m['MAE']:7.2f}  R2 {m['R2']:6.3f}")
    out = data_path.replace(".npz", f"_{model}.json")
    json.dump(res, open(out, "w"), indent=2)
    print("saved ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["timesfm", "moirai"], required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    main(args.model, args.data)
