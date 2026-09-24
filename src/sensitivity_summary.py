"""Summarise sensitivity_*_seed*.json (from eval_sensitivity.py) as mean +- sd over seeds.

Prints, for each model and each shifted setting, the overall / process-node /
gate RMSE (mean +- sd over the seeds found) and the relative change of the
overall RMSE with respect to the baseline test split, plus LaTeX rows.

Usage: python sensitivity_summary.py <results_dir> [stem]   (stem default tensors_mco_1h_h3)
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

res_dir = sys.argv[1]
stem = sys.argv[2] if len(sys.argv) > 2 else "tensors_mco_1h_h3"
files = sorted(glob.glob(os.path.join(res_dir, f"sensitivity_{stem}_seed*.json")))
print("files:", [os.path.basename(f) for f in files])
acc = defaultdict(lambda: defaultdict(list))   # model -> setting -> [(overall, process, gates)]
order = []
for f in files:
    d = json.load(open(f))
    for model, settings in d.items():
        if model.startswith("_"):
            continue
        for s, r in settings.items():
            if s not in order:
                order.append(s)
            acc[model][s].append((r["overall"]["RMSE"], r["checkpoints"]["RMSE"], r["gates"]["RMSE"]))

def ms(v):
    a = np.asarray(v)
    return a.mean(axis=0), a.std(axis=0), len(a)

print(f"\n{'model':12s} {'setting':14s} {'overall':>16s} {'process':>16s} {'gates':>14s} {'vs base':>8s}  n")
latex = []
for model in acc:
    base = ms(acc[model]["base_test"])[0]
    for s in order:
        if s not in acc[model]:
            continue
        m, sd, n = ms(acc[model][s])
        rel = 100 * (m[0] / base[0] - 1)
        print(f"{model:12s} {s:14s} {m[0]:7.1f}+-{sd[0]:4.1f}   {m[1]:7.1f}+-{sd[1]:4.1f}   {m[2]:6.1f}+-{sd[2]:4.1f}   {rel:+6.1f}%  {n}")
        latex.append(f"{model} & {s} & {m[0]:.1f} $\\pm$ {sd[0]:.1f} & {m[1]:.1f} & {m[2]:.1f} & {rel:+.0f}\\% \\\\")
print("\nLaTeX rows:")
print("\n".join(latex))
