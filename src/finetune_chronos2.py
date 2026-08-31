"""LoRA fine-tuning of Chronos-2 on the airport dataset ("rung 3" baseline).

Fairness ladder this completes:
  1. Chronos-Bolt   zero-shot, no covariates      (baseline_chronos.py)
  2. Chronos-2      zero-shot, schedule covariates (baseline_chronos2.py)
  3. Chronos-2+LoRA fine-tuned, schedule covariates  <-- THIS SCRIPT
  4. GCN-LSTM+Sched supervised                     (train_models.py)

Rung 3 vs 4 isolates ARCHITECTURE: both are trained on the same chronological
70% train split, validated on the same 15%, tested on the same 15% windows,
and both receive the same schedule covariates.

Uses Chronos2Pipeline.fit() (built into chronos-forecasting >= 2.0):
LoRA mode targets the attention projections + output head; recommended
lr 1e-5; validation_inputs enables best-checkpoint selection on eval_loss.

GPU strongly recommended (Colab T4 is fine). CPU works for --probe only.

Usage (Colab):
  pip install "chronos-forecasting>=2.0" peft
  python finetune_chronos2.py --data tensors_1h_h3.npz            # full run
  python finetune_chronos2.py --data tensors_1h_h3.npz --probe    # 5-step smoke test
"""
import argparse
import json
import numpy as np
import torch

from airport_graph import NODES, CHECKPOINTS
from train_models import metrics
from baseline_chronos2 import reconstruct_series, SCHED_NAMES

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15   # must match train_models.chrono_split


def build_series(data_path):
    d = np.load(data_path, allow_pickle=True)
    X, Y, S = d["X"], d["Y"], d["S"]
    feats = list(d["features"])
    n, in_seq, horizon = X.shape[0], X.shape[3], Y.shape[2]

    tgt = reconstruct_series(X, 0)                       # (T_hist, nodes) Flow_In
    sched_idx = [feats.index(nm) for nm in SCHED_NAMES]
    sched = {nm: reconstruct_series(X, i) for nm, i in zip(SCHED_NAMES, sched_idx)}
    return d, X, Y, S, tgt, sched, n, in_seq, horizon


def fit_inputs(tgt, sched, t_end, horizon, use_cov=True):
    """One training dict per node, target/covariates truncated at time t_end.

    Format follows Chronos2Pipeline.fit == predict:
      - target & past_covariates: aligned, length t_end
      - future_covariates: the next `horizon` values of each schedule series
        (they are known-future by construction). Including this key also
        signals to the fit() dataset builder which covariates are available
        into the future.
    """
    out = []
    for node in range(tgt.shape[1]):
        item = {"target": tgt[:t_end, node].astype(np.float32)}
        if use_cov:
            item["past_covariates"] = {nm: sched[nm][:t_end, node]
                                       .astype(np.float32) for nm in SCHED_NAMES}
            item["future_covariates"] = {nm: sched[nm][t_end:t_end + horizon, node]
                                         .astype(np.float32) for nm in SCHED_NAMES}
        out.append(item)
    return out


def val_inputs(tgt, sched, val_ends, horizon, use_cov=True):
    """Validation dicts at several forecast origins inside the val range.
    Chronos2Dataset VALIDATION mode holds out the final prediction_length
    steps of each target as ground truth."""
    out = []
    for t_end in val_ends:
        for node in range(tgt.shape[1]):
            item = {"target": tgt[:t_end + horizon, node].astype(np.float32)}
            if use_cov:
                item["past_covariates"] = {nm: sched[nm][:t_end + horizon, node]
                                           .astype(np.float32)
                                           for nm in SCHED_NAMES}
                item["future_covariates"] = {nm: sched[nm][t_end + horizon:
                                                           t_end + 2 * horizon,
                                                           node]
                                             .astype(np.float32)
                                             for nm in SCHED_NAMES}
            out.append(item)
    return out


def evaluate(pipe, X, Y, S, tgt, sched, n, in_seq, horizon,
             context_len, tag, data_path, batch_origins=16, max_origins=None,
             use_cov=True, out_suffix="_chronos2_lora"):
    """Identical evaluation loop to baseline_chronos2.py: same test windows,
    same metrics, same per-scope breakdown."""
    test_start = int(n * (TRAIN_FRAC + VAL_FRAC))
    if max_origins:                       # --probe: truncated smoke evaluation
        n = min(n, test_start + max_origins)
    preds = np.zeros((n - test_start, len(NODES), horizon), dtype=np.float32)
    batch, slots = [], []

    def flush():
        nonlocal batch, slots
        if not batch:
            return
        out = pipe.predict_quantiles(batch, prediction_length=horizon,
                                     quantile_levels=[0.5])
        mean = out[1] if isinstance(out, tuple) else out
        for (kk, node), m in zip(slots, mean):
            preds[kk, node] = np.asarray(m).reshape(-1)[:horizon]
        batch, slots = [], []

    for k, s in enumerate(range(test_start, n)):
        end = s + in_seq
        lo = max(0, end - context_len)
        for node in range(len(NODES)):
            item = {"target": tgt[lo:end, node]}
            if use_cov:
                item["past_covariates"] = {nm: sched[nm][lo:end, node]
                                           for nm in SCHED_NAMES}
                item["future_covariates"] = {nm: S[s, node, j, :]
                                             for j, nm in enumerate(SCHED_NAMES)}
            batch.append(item)
            slots.append((k, node))
        if len(batch) >= batch_origins * len(NODES):
            flush()
        if (k + 1) % 32 == 0:
            print(f"  {k + 1}/{n - test_start} origins")
    flush()

    y_true = Y[test_start:n]
    preds = np.maximum(preds, 0)
    cp_mask = np.array([nm in CHECKPOINTS for nm in NODES])
    res = {"overall": metrics(y_true, preds),
           "checkpoints": metrics(y_true, preds, cp_mask),
           "gates": metrics(y_true, preds, ~cp_mask)}
    print(f"\n== {tag} ==")
    for scope, m in res.items():
        print(f"  {scope:12s} RMSE {m['RMSE']:8.2f}  MAE {m['MAE']:7.2f}  "
              f"R2 {m['R2']:6.3f}  MAPE(nz) {m['MAPE_nonzero']:7.1f}%")
    out = data_path.replace(".npz", out_suffix + ".json")
    json.dump(res, open(out, "w"), indent=2)
    np.savez_compressed(data_path.replace(".npz", out_suffix + "_preds.npz"),
                        y_true=y_true, pred_Chronos2FT=preds)
    print(f"saved -> {out}")
    return res


def main(args):
    d, X, Y, S, tgt, sched, n, in_seq, horizon = build_series(args.data)
    train_end_w = int(n * TRAIN_FRAC)          # window index
    val_end_w = int(n * (TRAIN_FRAC + VAL_FRAC))
    # Last time index any TRAIN window's target reaches — the supervised
    # models saw exactly this much of the series, so Chronos-2 gets the same.
    t_train = train_end_w - 1 + in_seq + horizon
    # Validation forecast origins: spread through the val range, history end
    # = window_idx + in_seq, never touching the test range.
    n_vo = 4 if args.probe else args.val_origins
    val_ws = np.linspace(train_end_w, val_end_w - 1, n_vo).astype(int)
    val_ends = [w + in_seq for w in val_ws]

    print(f"windows: train<{train_end_w} val<{val_end_w} test<{n} | "
          f"train series len {t_train} | {len(val_ends)} val origins x {len(NODES)} nodes")

    from chronos import Chronos2Pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2", device_map=device,
        torch_dtype=torch.float32 if device == "cpu" else "auto")

    use_cov = not args.no_covariates
    num_steps = 5 if args.probe else args.steps
    ft = pipe.fit(
        fit_inputs(tgt, sched, t_train, horizon, use_cov),
        prediction_length=horizon,
        validation_inputs=val_inputs(tgt, sched, val_ends, horizon, use_cov),
        finetune_mode=args.mode,
        learning_rate=args.lr,
        num_steps=num_steps,
        batch_size=args.batch_size,
        context_length=args.context,
        output_dir=args.out_dir,
    )
    # fit() returns the fine-tuned pipeline in current versions; fall back to
    # loading the auto-saved checkpoint if the API returns something else.
    if not hasattr(ft, "predict_quantiles"):
        import glob
        ckpts = sorted(glob.glob(f"{args.out_dir}/**/finetuned-ckpt",
                                 recursive=True))
        assert ckpts, f"no checkpoint found under {args.out_dir}"
        ft = Chronos2Pipeline.from_pretrained(ckpts[-1], device_map=device)

    cov_tag = "schedule covariates" if use_cov else "NO covariates (ablation)"
    tag = f"Chronos-2 + {args.mode.upper()} fine-tune ({num_steps} steps, {cov_tag})"
    suffix = f"_chronos2_{args.mode}" + ("" if use_cov else "_nocov")
    evaluate(ft, X, Y, S, tgt, sched, n, in_seq, horizon,
             args.context, tag, args.data,
             max_origins=8 if args.probe else None,
             use_cov=use_cov, out_suffix=suffix)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", default="lora", choices=["lora", "full"])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="1e-5 for LoRA (default), 1e-6 for full")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="lower than the 256 default: covariates raise the "
                    "per-series cost, and the dataset is small")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--val-origins", type=int, default=8)
    ap.add_argument("--out-dir", default="chronos2_finetune")
    ap.add_argument("--probe", action="store_true",
                    help="5-step smoke test of the whole path")
    ap.add_argument("--no-covariates", action="store_true",
                    help="ablation: fine-tune and evaluate WITHOUT the "
                    "schedule covariates (isolates adaptation from covariates)")
    args = ap.parse_args()
    main(args)
