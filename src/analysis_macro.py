"""Process-node-centred summary of every results file.

For each model in each *_results*.json it prints
    overall RMSE | process-node (checkpoints) RMSE | gate RMSE | MACRO RMSE
where MACRO = (process-node RMSE + gate RMSE) / 2, the equal-weighted metric
that stops the 24 near-deterministic gate nodes from dominating the pooled
"overall" number (8 of 32 nodes are process nodes).

Seeds are pooled automatically: files that differ only by the _seedN suffix
are averaged and reported as mean +- sd (n seeds).

Usage:
    python analysis_macro.py <results_dir> [stem_filter ...]
    python analysis_macro.py results mco_1h_h3 exppax
    python analysis_macro.py results --latex   # LaTeX rows
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

SCOPES = ("overall", "checkpoints", "gates")


def load_dir(res_dir):
    """stem -> model -> list of (overall, process, gates, macro) over seeds."""
    table = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(res_dir, "*.json"))):
        base = os.path.basename(f)[:-5]
        m = re.match(r"(.*?)(?:_results|_chronos2_lora_nocov|_chronos2_lora|_chronos2_nocov|_chronos2|_chronos)(?:_seed(\d+))?$", base)
        if not m:
            continue
        stem = base.replace(f"_seed{m.group(2)}", "") if m.group(2) else base
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        # single-model files (Chronos baselines) store the scopes at top level
        if set(SCOPES) <= set(d):
            tag = base.split("_chronos", 1)[1] if "_chronos" in base else "model"
            names = {"": "Chronos-Bolt ZS", "2": "Chronos-2 ZS+cov", "2_nocov": "Chronos-2 ZS no-cov",
                     "2_lora": "Chronos-2 LoRA+cov", "2_lora_nocov": "Chronos-2 LoRA no-cov"}
            stem = base.split("_chronos", 1)[0] + "_results" if "_chronos" in base else stem
            d = {names.get(tag, "Chronos " + tag): d}
        for model, r in d.items():
            if not (isinstance(r, dict) and set(SCOPES) <= set(r)):
                continue
            o, p, g = (r[s]["RMSE"] for s in SCOPES)
            table[stem][model].append((o, p, g, (p + g) / 2))
    return table


def fmt(vals):
    a = np.asarray(vals)
    if len(a) == 1:
        return f"{a[0]:6.1f}      "
    return f"{a.mean():6.1f}+-{a.std(ddof=0):4.1f}"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    latex = "--latex" in sys.argv
    res_dir, filters = args[0], args[1:]
    table = load_dir(res_dir)
    for stem in sorted(table):
        if filters and not any(f in stem for f in filters):
            continue
        print(f"\n== {stem} ==")
        if latex:
            print("model & overall & process & gates & macro \\\\")
        else:
            print(f"{'model':28s} {'overall':>13s} {'process':>13s} {'gates':>13s} {'MACRO':>13s}  n")
        rows = table[stem]
        for model in sorted(rows, key=lambda k: np.mean([v[3] for v in rows[k]])):
            v = np.array(rows[model])
            if latex:
                cells = " & ".join(f"{v[:, i].mean():.1f}" + (f"$\\pm${v[:, i].std():.1f}" if len(v) > 1 else "")
                                   for i in range(4))
                print(f"{model} & {cells} \\\\")
            else:
                print(f"{model:28s} {fmt(v[:, 0])} {fmt(v[:, 1])} {fmt(v[:, 2])} {fmt(v[:, 3])}  {len(v)}")


if __name__ == "__main__":
    main()
