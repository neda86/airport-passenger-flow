"""Bootstrap confidence intervals across TEST DAYS for every model in a preds file.

Forecast origins are grouped by calendar day (from the timestamps stored in
the npz); days are resampled with replacement B times and the RMSE is
recomputed for each resample, so the interval reflects day-to-day variability
of the test period rather than the seed. Reported for the pooled ("overall"),
process-node and gate scopes and for the macro metric
(process + gate) / 2. With --ref MODEL, paired differences MODEL_i - REF on
the same resampled days are also reported (a CI excluding 0 means the two
models differ at the 95% level on this test period).

Usage:
    AIRPORT_LAYOUT=mco python bootstrap_ci.py data/tensors_mco_1h_h3_preds.npz \
        [more *_preds.npz] [--B 2000] [--ref "LSTM+Sched"] [--latex] [--out ci.json]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from airport_graph import NODES, CHECKPOINTS  # noqa: E402

CP = np.array([n in CHECKPOINTS for n in NODES])


def rmse_scopes(y, p):
    e2 = (p - y) ** 2
    o = np.sqrt(e2.mean())
    c = np.sqrt(e2[:, CP, :].mean())
    g = np.sqrt(e2[:, ~CP, :].mean())
    return np.array([o, c, g, (c + g) / 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    y_true, stamps, preds = None, None, {}
    for f in a.preds:
        d = np.load(f, allow_pickle=True)
        if y_true is None:
            y_true = d["y_true"]
            if "timestamps" not in d.files:
                sys.exit("first file must contain timestamps (train_models.py output)")
            stamps = d["timestamps"]
        elif d["y_true"].shape != y_true.shape or not np.allclose(d["y_true"], y_true, atol=1e-3):
            print(f"warning: {f} has a different y_true; skipped")
            continue
        for k in d.files:
            if k.startswith("pred_"):
                preds[k[5:]] = np.maximum(d[k], 0)

    days = np.asarray(stamps).astype("datetime64[D]")
    uniq, day_idx = np.unique(days, return_inverse=True)
    groups = [np.where(day_idx == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(a.seed)
    names = list(preds)
    point = {n: rmse_scopes(y_true, preds[n]) for n in names}
    boot = {n: np.zeros((a.B, 4)) for n in names}
    for b in range(a.B):
        pick = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[i] for i in pick])
        yb = y_true[rows]
        for n in names:
            boot[n][b] = rmse_scopes(yb, preds[n][rows])

    scopes = ["overall", "process", "gates", "macro"]
    res = {"_config": {"B": a.B, "n_test_days": int(len(uniq)), "n_origins": int(len(days))}}
    print(f"{len(uniq)} test days, {len(days)} origins, B={a.B}")
    hdr = f"{'model':28s} " + " ".join(f"{s:>22s}" for s in scopes)
    print(hdr if not a.latex else "model & " + " & ".join(scopes) + " \\\\")
    for n in sorted(names, key=lambda k: point[k][3]):
        lo, hi = np.percentile(boot[n], [2.5, 97.5], axis=0)
        res[n] = {s: {"point": float(point[n][i]), "ci95": [float(lo[i]), float(hi[i])]}
                  for i, s in enumerate(scopes)}
        cells = [f"{point[n][i]:.1f} [{lo[i]:.1f}, {hi[i]:.1f}]" for i in range(4)]
        print((f"{n:28s} " + " ".join(f"{c:>22s}" for c in cells)) if not a.latex
              else f"{n} & " + " & ".join(cells) + " \\\\")

    if a.ref and a.ref in boot:
        print(f"\npaired differences vs {a.ref} (negative = better than reference)")
        for n in sorted(names, key=lambda k: point[k][3]):
            if n == a.ref:
                continue
            diff = boot[n] - boot[a.ref]
            lo, hi = np.percentile(diff, [2.5, 97.5], axis=0)
            pt = point[n] - point[a.ref]
            res[n]["diff_vs_ref"] = {s: {"point": float(pt[i]), "ci95": [float(lo[i]), float(hi[i])]}
                                     for i, s in enumerate(scopes)}
            cells = [f"{pt[i]:+.1f} [{lo[i]:+.1f}, {hi[i]:+.1f}]" for i in range(4)]
            print(f"{n:28s} " + " ".join(f"{c:>22s}" for c in cells))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        print("saved ->", a.out)


if __name__ == "__main__":
    main()
