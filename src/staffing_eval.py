"""Operational decision experiment: predicted checkpoint inflow -> screening lanes.

For every forecast origin, checkpoint and horizon step, the number of lanes an
operator would open is

    lanes = ceil(passengers per hour / passengers served per lane per hour)

computed once from the TRUE inflow and once from each model's forecast. The
simulator's standard lane serves one passenger every 32 s on average
(SEC_SVC in generate_mco.py), i.e. 112.5 passengers per lane-hour; the
PreCheck lane (20 s) is ignored so that all models are scored with the same
rule. Metrics per model (pooled over both checkpoints and all horizon steps
unless --step is given):

    understaffed hours   share of checkpoint-hours where predicted lanes < needed lanes
    surplus lane-hours   mean of max(predicted - needed, 0) per checkpoint-hour
    peak recall          share of peak hours (needed lanes >= --peak, default the
                         90th percentile of needed lanes) that the forecast also
                         flags as peak
    staffing MAE         mean |predicted - needed| lanes

Reads the *_preds.npz written by train_models.py / baseline_chronos2.py /
finetune_chronos2.py (y_true plus pred_<model> arrays of shape
(origins, nodes, horizon)). Several files can be given; models are merged.

Usage:
    AIRPORT_LAYOUT=mco python staffing_eval.py data/tensors_mco_1h_h3_preds.npz \
        data/tensors_mco_1h_h3_tdnabl_preds.npz data/tensors_mco_1h_h3_chronos2_lora_preds.npz \
        [--rate 112.5] [--step 3] [--peak 6] [--freq-min 60] [--latex] [--out staffing.json]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from airport_graph import NODES, CHECKPOINTS  # noqa: E402

SECURITY_NODES = [n for n in CHECKPOINTS if n.startswith("Security")]


def lanes_needed(pax_per_bin, rate_per_hour, freq_min):
    per_hour = pax_per_bin * (60.0 / freq_min)
    return np.ceil(np.maximum(per_hour, 0) / rate_per_hour)


def score(true_l, pred_l, peak):
    under = float(np.mean(pred_l < true_l))
    surplus = float(np.mean(np.maximum(pred_l - true_l, 0)))
    mae = float(np.mean(np.abs(pred_l - true_l)))
    is_peak = true_l >= peak
    recall = float(np.mean(pred_l[is_peak] >= peak)) if is_peak.any() else float("nan")
    return {"understaffed_share": under, "surplus_lane_hours": surplus,
            "staffing_MAE": mae, "peak_recall": recall,
            "n_hours": int(true_l.size), "n_peak_hours": int(is_peak.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--rate", type=float, default=3600.0 / 32.0, help="pax per lane-hour")
    ap.add_argument("--freq-min", type=float, default=60.0, help="bin width of the tensors")
    ap.add_argument("--step", type=int, default=None, help="1-based horizon step (default: all pooled)")
    ap.add_argument("--peak", type=float, default=None, help="lanes that define a peak hour")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    idx = [NODES.index(n) for n in SECURITY_NODES]
    y_true, preds = None, {}
    for f in a.preds:
        d = np.load(f, allow_pickle=True)
        yt = d["y_true"][:, idx, :]
        if y_true is None:
            y_true = yt
        elif yt.shape != y_true.shape or not np.allclose(yt, y_true, atol=1e-3):
            print(f"warning: y_true in {f} differs from the first file (different tensor?); skipped")
            continue
        for k in d.files:
            if k.startswith("pred_"):
                preds[k[5:]] = d[k][:, idx, :]
    if a.step:
        y_true = y_true[:, :, a.step - 1:a.step]
        preds = {k: v[:, :, a.step - 1:a.step] for k, v in preds.items()}

    true_l = lanes_needed(y_true, a.rate, a.freq_min)
    peak = a.peak if a.peak is not None else float(np.percentile(true_l, 90))
    print(f"checkpoints {SECURITY_NODES}; rate {a.rate:.1f} pax/lane-h; "
          f"needed lanes: mean {true_l.mean():.2f}, max {true_l.max():.0f}; peak threshold >= {peak:.0f} lanes")

    res = {"_config": {"rate_per_lane_hour": a.rate, "peak_lanes": peak, "step": a.step,
                       "nodes": SECURITY_NODES}}
    res["Oracle (true inflow)"] = score(true_l, true_l, peak)
    for name, p in preds.items():
        res[name] = score(true_l, lanes_needed(p, a.rate, a.freq_min), peak)

    rows = [(k, v) for k, v in res.items() if not k.startswith("_")]
    rows.sort(key=lambda kv: kv[1]["staffing_MAE"])
    if a.latex:
        print("model & understaffed (\\%) & surplus lane-h & peak recall (\\%) & staffing MAE \\\\")
        for k, v in rows:
            print(f"{k} & {100 * v['understaffed_share']:.1f} & {v['surplus_lane_hours']:.2f} & "
                  f"{100 * v['peak_recall']:.1f} & {v['staffing_MAE']:.2f} \\\\")
    else:
        print(f"\n{'model':28s} {'under%':>7s} {'surplus':>8s} {'peakRec%':>9s} {'MAE':>6s}")
        for k, v in rows:
            print(f"{k:28s} {100 * v['understaffed_share']:7.1f} {v['surplus_lane_hours']:8.2f} "
                  f"{100 * v['peak_recall']:9.1f} {v['staffing_MAE']:6.2f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        print("saved ->", a.out)


if __name__ == "__main__":
    main()
