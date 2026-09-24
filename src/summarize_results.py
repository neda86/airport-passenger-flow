"""Compact summary of every *_results*.json / *_chronos*.json in a results folder.
Prints one line per (stem, model): RMSE mean+-sd over seeds, MAE, R2, checkpoint RMSE,
gate RMSE. Designed to be read from a Colab cell output.
    python summarize_results.py results [stem-filter]
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "results"
flt = sys.argv[2] if len(sys.argv) > 2 else ""
runs = defaultdict(lambda: defaultdict(list))          # stem -> model -> [metrics per seed]
for f in sorted(glob.glob(os.path.join(root, "*.json"))):
    name = os.path.basename(f)[:-5]
    if flt and flt not in name:
        continue
    if "multiseed" in name:
        continue
    m = re.match(r"(.*?)_(results|chronos2_lora_nocov|chronos2_lora|chronos2|chronos)(_seed\d+)?$", name)
    if not m:
        continue
    stem, kind = m.group(1), m.group(2)
    try:
        d = json.load(open(f))
    except Exception as e:                                # noqa: BLE001
        print("skip", name, e); continue
    if kind == "results":
        for model, r in d.items():
            runs[stem][model].append(r)
    else:
        label = {"chronos": "Chronos-Bolt ZS", "chronos2": "Chronos-2 ZS+cov",
                 "chronos2_lora": "Chronos-2 LoRA+cov", "chronos2_lora_nocov": "Chronos-2 LoRA nocov"}[kind]
        r = d if "overall" in d else next(iter(d.values()))
        runs[stem][label].append(r)


def g(r, scope, key):
    try:
        return float(r[scope][key])
    except Exception:                                     # noqa: BLE001
        return float("nan")


for stem in sorted(runs):
    print(f"\n## {stem}")
    print(f"{'model':24s} {'n':>2s} {'RMSE':>12s} {'MAE':>6s} {'R2':>6s} {'cpRMSE':>7s} {'gtRMSE':>7s}")
    for model, rs in sorted(runs[stem].items(), key=lambda kv: np.nanmean([g(r, 'overall', 'RMSE') for r in kv[1]])):
        rm = np.array([g(r, "overall", "RMSE") for r in rs])
        line = (f"{model:24s} {len(rs):2d} {np.nanmean(rm):6.1f}+-{np.nanstd(rm):4.1f} "
                f"{np.nanmean([g(r,'overall','MAE') for r in rs]):6.1f} {np.nanmean([g(r,'overall','R2') for r in rs]):6.3f} "
                f"{np.nanmean([g(r,'checkpoints','RMSE') for r in rs]):7.1f} {np.nanmean([g(r,'gates','RMSE') for r in rs]):7.1f}")
        print(line)
