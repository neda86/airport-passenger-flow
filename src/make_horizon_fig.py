"""Horizon-sweep figure for the AIAA paper from a small JSON of numbers
({model: {horizon: [overall, process, gates]}}), produced by summarize_results.py output.
    python make_horizon_fig.py --numbers horizon_numbers.json --out figures
"""
import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5, "pdf.fonttype": 42,
})
STYLE = {
    "MovingAverage": dict(color="#7f7f7f", ls="--", marker="o", label="Moving average"),
    "LSTM": dict(color="#1f77b4", ls="--", marker="o", label="LSTM (history)"),
    "TDN": dict(color="#d62728", ls="--", marker="s", label="TDN (history)"),
    "LSTM+Sched": dict(color="#1f77b4", ls="-", marker="o", label="LSTM + schedule"),
    "GCN-LSTM+Sched": dict(color="#2ca02c", ls="-", marker="^", label="GCN-LSTM + schedule"),
    "TDN+Sched": dict(color="#d62728", ls="-", marker="s", label="TDN + schedule"),
    "TDN hybrid+Sched": dict(color="#8c564b", ls="-", marker="*", label="TDN hybrid (token + prior + head)"),
    "Chronos-Bolt ZS": dict(color="#9467bd", ls=":", marker="D", label="Chronos-Bolt zero-shot"),
    "Chronos-2 ZS+cov": dict(color="#ff7f0e", ls="-.", marker="D", label="Chronos-2 zero-shot + covariates"),
    "LSTM ctx+Sched": dict(color="#e377c2", ls="-", marker="v", label="LSTM + schedule (context + prior)"),
    "LSTM ctxboth+Sched": dict(color="#000000", ls="-", marker="P", label="LSTM + schedule (context + prior + head)"),
}


def main(numbers, out):
    d = json.load(open(numbers))
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3), sharex=True)
    titles = ["All nodes", "Process nodes", "Gates"]
    for k, ax in enumerate(axes):
        for m, st in STYLE.items():
            if m not in d:
                continue
            hs = sorted(int(h) for h in d[m])
            ax.plot(hs, [d[m][str(h)][k] for h in hs], lw=1.1, ms=3.5, **st)
        ax.set_title(titles[k]); ax.set_xscale("log", base=2); ax.set_xticks([1, 3, 6, 12, 24]); ax.set_xticklabels(["1", "3", "6", "12", "24"])
        ax.grid(alpha=0.3); ax.set_xlabel("Forecast horizon (h)")
    axes[0].set_ylabel("Test RMSE (passengers per hour)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.24))
    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "horizon_sweep_1h.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, "horizon_sweep_1h.png"), dpi=200, bbox_inches="tight")
    print("wrote", os.path.join(out, "horizon_sweep_1h.pdf"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers", required=True); ap.add_argument("--out", default="figures")
    a = ap.parse_args(); main(a.numbers, a.out)
