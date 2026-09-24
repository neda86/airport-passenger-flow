"""Reproducibility table: parameter counts, configuration and inference time.

Instantiates every model configuration reported in the paper with the tensor
shapes of a given dataset, counts trainable parameters, and times one forward
pass over the test split. Training hyper-parameters are read from
train_models.py (epochs, patience, learning rate, batch size). Chronos-2
settings are read from finetune_chronos2.py defaults and, if the chronos
package is installed, the LoRA defaults of Chronos2Pipeline.fit are printed.

Usage:
    AIRPORT_LAYOUT=mco python model_card.py --data data/tensors_mco_1h_h3.npz [--latex] [--out model_card.json]
"""
import argparse
import inspect
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from airport_graph import adjacency, NODES  # noqa: E402
import train_models as tm  # noqa: E402

CONFIGS = [  # (paper name, spatial, temporal, uses schedule)
    ("LSTM", "none", "lstm", False), ("LSTM+Sched", "none", "lstm", True),
    ("GCN-LSTM", "gcn", "lstm", False), ("GCN-LSTM+Sched", "gcn", "lstm", True),
    ("GAT-LSTM+Sched", "gat", "lstm", True),
    ("Transformer", "none", "transformer", False), ("Transformer+Sched", "none", "transformer", True),
    ("TDN", "none", "tdn", False), ("TDN+Sched", "none", "tdn", True),
    ("GCN-TDN+Sched", "gcn", "tdn", True), ("GAT-TDN+Sched", "gat", "tdn", True),
    ("Hybrid (tdn_both)", "none", "tdn_both", True),
    ("Hybrid last-step (tdn_noq_both)", "none", "tdn_noq_both", True),
    ("LSTM with TDN pathway (lstm_ctx)", "none", "lstm_ctx", True),
    ("Iso-TDN+Sched", "isolated", "tdn", True), ("Diff-TDN+Sched", "diffusion", "tdn", True),
    ("Cheb-TDN+Sched", "cheb", "tdn", True),
]


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main(a):
    d = np.load(a.data, allow_pickle=True)
    X, Y, S = d["X"], d["Y"], d["S"]
    horizon, n_feat, sched_k = Y.shape[2], X.shape[2], S.shape[2]
    A = adjacency()
    raw = tm.chrono_split(X, S, Y)
    splits, _ = tm.normalize_with_train_stats(raw)
    (_, _, test_dl) = tm.loaders(splits)
    n_test = raw[2][0].shape[0]

    sig = inspect.signature(tm.train)
    card = {"dataset": os.path.basename(a.data),
            "tensor_shapes": {"X": list(X.shape), "Y": list(Y.shape), "S": list(S.shape)},
            "nodes": len(NODES), "horizon": horizon, "n_features": n_feat, "n_sched_covariates": sched_k,
            "split": "70/15/15 chronological (chrono_split)",
            "normalisation": "feature z-score and target min-max fit on the training split",
            "training": {"optimizer": "Adam", "lr": sig.parameters["lr"].default,
                         "early_stopping_patience": sig.parameters["patience"].default,
                         "max_epochs": 150, "batch_size": 32, "loss": "MSE on normalised target",
                         "seeds": [42, 7, 123]},
            "hardware": {"device": tm.DEVICE, "torch": torch.__version__, "python": platform.python_version(),
                         "cpu": platform.processor() or platform.machine(),
                         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
            "models": {}}
    print(f"{'model':36s} {'params':>9s} {'infer ms/origin':>16s}")
    for name, sp, tp, use_s in CONFIGS:
        try:
            m = tm.STModel(A, len(NODES), n_feat, horizon, sched_k=sched_k if use_s else 0,
                           spatial=sp, temporal=tp).to(tm.DEVICE).eval()
        except Exception as e:  # variant not present in this train_models version
            print(f"{name:36s} skipped ({e.__class__.__name__})")
            continue
        t0 = time.perf_counter()
        with torch.no_grad():
            tm.predict(m, test_dl)
        ms = 1000 * (time.perf_counter() - t0) / n_test
        card["models"][name] = {"spatial": sp, "temporal": tp, "schedule": use_s,
                                "trainable_params": n_params(m), "inference_ms_per_origin": round(ms, 3)}
        print(f"{name:36s} {n_params(m):9,d} {ms:16.2f}")

    card["chronos2"] = {"model": "amazon/chronos-2", "context_length": 512, "finetune_mode": "lora",
                        "steps": 1000, "lr": 1e-5, "batch_size": 32,
                        "covariates": ["sched_board", "sched_dep"], "note": "see finetune_chronos2.py"}
    try:
        from chronos import Chronos2Pipeline
        fsig = inspect.signature(Chronos2Pipeline.fit)
        lora = {k: v.default for k, v in fsig.parameters.items() if "lora" in k.lower()}
        card["chronos2"]["fit_signature_lora_defaults"] = {k: str(v) for k, v in lora.items()}
        print("Chronos2Pipeline.fit LoRA defaults:", lora or "(not exposed in signature; see chronos source)")
    except Exception as e:
        print("chronos package not available here:", e.__class__.__name__)

    if a.latex:
        print("\nmodel & trainable parameters & inference (ms/origin) \\\\")
        for k, v in card["models"].items():
            print(f"{k} & {v['trainable_params']:,} & {v['inference_ms_per_origin']:.2f} \\\\")
    out = a.out or a.data.replace(".npz", "_model_card.json")
    json.dump(card, open(out, "w"), indent=2)
    print("saved ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--out", default=None)
    main(ap.parse_args())
