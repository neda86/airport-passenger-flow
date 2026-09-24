"""Aggregate per-seed result files into mean +/- std tables.

train_models.py writes <stem>_results.json (seed 42), <stem>_results_seed7.json,
<stem>_results_seed123.json, ... each of the form
    {model: {overall|checkpoints|gates: {RMSE, MAE, R2, MAPE_nonzero}}}

This script collects every file matching <stem>_results*.json, and writes
<stem>_multiseed.json in the same format as results/multiseed_*.json
    {model: {scope: {metric: {mean, std, n}}}}
and prints a markdown table (RMSE / MAE / R2, overall + checkpoints + gates).

Usage: aggregate_seeds.py --dir results --stem tensors_mco_1h_h3 [--latex]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

SCOPES = ["overall", "checkpoints", "gates"]
METRICS = ["RMSE", "MAE", "R2"]


def load(dir_, stem):
    files = sorted(glob.glob(os.path.join(dir_, f"{stem}_results*.json")))
    runs = {}
    for f in files:
        m = re.search(r"_results(?:_seed(\d+))?\.json$", f)
        seed = int(m.group(1)) if m and m.group(1) else 42
        runs[seed] = json.load(open(f))
    return runs


def aggregate(runs):
    models = list(next(iter(runs.values())).keys())
    out = {}
    for mdl in models:
        out[mdl] = {}
        for sc in SCOPES:
            out[mdl][sc] = {}
            for met in METRICS:
                vals = [r[mdl][sc][met] for r in runs.values() if mdl in r]
                out[mdl][sc][met] = {"mean": float(np.mean(vals)),
                                     "std": float(np.std(vals, ddof=0)),
                                     "n": len(vals)}
    return out


def fmt(cell, met):
    m, s = cell["mean"], cell["std"]
    return f"{m:.3f} ± {s:.3f}" if met == "R2" else f"{m:.1f} ± {s:.1f}"


def markdown(agg):
    hdr = "| Model | " + " | ".join(f"{sc} {met}" for sc in SCOPES for met in METRICS) + " |"
    sep = "|---|" + "---|" * (len(SCOPES) * len(METRICS))
    rows = [hdr, sep]
    for mdl, d in agg.items():
        rows.append(f"| {mdl} | " + " | ".join(fmt(d[sc][met], met) for sc in SCOPES for met in METRICS) + " |")
    return "\n".join(rows)


def latex(agg):
    """Rows for a booktabs table: model & RMSE & MAE & R2 (overall) & RMSE (cp) & RMSE (gates)."""
    rows = []
    for mdl, d in agg.items():
        o, c, g = d["overall"], d["checkpoints"], d["gates"]
        rows.append(f"{mdl} & {o['RMSE']['mean']:.1f} $\\pm$ {o['RMSE']['std']:.1f} & "
                    f"{o['MAE']['mean']:.1f} $\\pm$ {o['MAE']['std']:.1f} & "
                    f"{o['R2']['mean']:.3f} & {c['RMSE']['mean']:.1f} & {g['RMSE']['mean']:.1f} \\\\")
    return "\n".join(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--stem", default="tensors_mco_1h_h3")
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    runs = load(a.dir, a.stem)
    if not runs:
        raise SystemExit(f"no {a.stem}_results*.json in {a.dir}")
    print(f"seeds: {sorted(runs)}")
    agg = aggregate(runs)
    out = os.path.join(a.dir, f"{a.stem}_multiseed.json")
    json.dump(agg, open(out, "w"), indent=2)
    print(markdown(agg))
    if a.latex:
        print()
        print(latex(agg))
    print(f"-> {out}")
