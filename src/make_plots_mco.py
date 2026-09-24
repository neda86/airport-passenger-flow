"""Publication figures for the MCO-layout experiments (vector PDF + PNG).

Works for any layout selected with AIRPORT_LAYOUT (node names, checkpoint
list and the 'security' node are taken from airport_graph). Reads the
predictions saved by train_models.py and produces, for a given tag:

  1. pred_vs_actual_<tag>      hexbin actual vs predicted, one panel per model
  2. timeseries_<tag>          4-day trace at a security checkpoint and the busiest gate
  3. rmse_by_horizon_<tag>     error growth over the forecast horizon
  4. rmse_by_node_<tag>        per-node RMSE: process nodes + 6 busiest gates
  5. schedule_source_<tag>     (optional) clean / published / realized / mixed
                               schedule sources, RMSE by node type

Usage:
  AIRPORT_LAYOUT=mco python make_plots_mco.py --preds data/tensors_mco_1h_h3_preds.npz --out figs_mco
  AIRPORT_LAYOUT=mco python make_plots_mco.py --preds data/tensors_mco_1h_h3_preds.npz --out figs_mco \
      --disruption clean=data/tensors_mco_1h_h3_results.json \
                   published=data_dis/tensors_mcodis_pub_1h_h3_results.json \
                   realized=data_dis/tensors_mcodis_real_1h_h3_results.json \
                   mixed=data_dis/tensors_mcodis_mix_1h_h3_results.json
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "pdf.fonttype": 42,
})

from airport_graph import NODES, CHECKPOINTS

# main comparison set (present in every preds file written by train_models.py)
MODELS = ["MovingAverage", "LSTM", "LSTM+Sched", "GCN-LSTM+Sched", "TDN+Sched"]
LABEL = {"MovingAverage": "Moving average", "LSTM": "LSTM (history)",
         "LSTM+Sched": "LSTM + schedule", "GCN-LSTM+Sched": "GCN-LSTM + schedule",
         "TDN+Sched": "TDN + schedule", "GCN-TDN+Sched": "GCN-TDN + schedule"}
COLOR = {"MovingAverage": "#7f7f7f", "LSTM": "#1f77b4", "LSTM+Sched": "#ff7f0e",
         "GCN-LSTM+Sched": "#2ca02c", "TDN+Sched": "#d62728", "GCN-TDN+Sched": "#9467bd"}


def rmse(a, b, axis=None):
    return np.sqrt(np.mean((a - b) ** 2, axis=axis))


def savefig(fig, out, name):
    fig.savefig(f"{out}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_pred_vs_actual(y, preds, out, tag, models=("LSTM", "LSTM+Sched", "TDN+Sched")):
    fig, axes = plt.subplots(1, len(models), figsize=(6.3, 2.4), sharex=True, sharey=True)
    lim = np.percentile(y, 99.8)
    for ax, m in zip(axes, models):
        p = preds[m]
        hb = ax.hexbin(y.ravel(), p.ravel(), gridsize=45, bins="log",
                       cmap="viridis", extent=(0, lim, 0, lim))
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        r2 = 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)
        ax.set_title(LABEL.get(m, m), fontsize=9)
        ax.text(0.05, 0.95, f"RMSE {rmse(y, p):.1f}\n$R^2$ {r2:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(fc="white", ec="0.6", alpha=0.9, boxstyle="round,pad=0.25"))
        ax.set_xlabel("Actual passenger flow")
    axes[0].set_ylabel("Predicted flow")
    fig.colorbar(hb, ax=axes, shrink=0.85, label="log count")
    savefig(fig, out, f"pred_vs_actual_{tag}")


def fig_timeseries(y, preds, stamps, out, tag, models=("LSTM", "LSTM+Sched", "TDN+Sched")):
    sec_name = next((n for n in NODES if n.startswith("Security")), CHECKPOINTS[-1])
    sec = NODES.index(sec_name)
    n_cp = len(CHECKPOINTS)
    busy_gate = n_cp + int(np.argmax(y[:, n_cp:, 0].sum(axis=0)))
    steps_per_day = 96 if "15min" in tag else 24
    span = slice(0, 4 * steps_per_day)
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 3.6), sharex=True)
    t = stamps[span]
    for ax, node in zip(axes, [sec, busy_gate]):
        ax.plot(t, y[span, node, 0], "k-", lw=1.5, label="Actual")
        for m in models:
            ax.plot(t, preds[m][span, node, 0], lw=1.0, alpha=0.9, color=COLOR[m], label=LABEL.get(m, m))
        ax.set_title(f"{NODES[node]}, one-step-ahead forecast", fontsize=9)
        ax.set_ylabel("Flow in")
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=4, loc="upper right", framealpha=0.92)
    axes[1].set_xlabel("Time")
    fig.autofmt_xdate()
    fig.tight_layout()
    savefig(fig, out, f"timeseries_{tag}")


def fig_rmse_by_horizon(y, preds, out, tag):
    H = y.shape[2]
    step_min = 15 if "15min" in tag else 60
    xs = np.arange(1, H + 1) * step_min / 60.0
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for m in MODELS:
        if m in preds:
            ax.plot(xs, rmse(y, preds[m], axis=(0, 1)), "o-", ms=3.5, color=COLOR[m], label=LABEL.get(m, m))
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("RMSE (passengers)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    savefig(fig, out, f"rmse_by_horizon_{tag}")


def fig_rmse_by_node(y, preds, out, tag, models=("LSTM", "LSTM+Sched", "TDN+Sched")):
    n_cp = len(CHECKPOINTS)
    totals = y.sum(axis=(0, 2))
    order = list(range(n_cp)) + list(n_cp + np.argsort(-totals[n_cp:])[:6])
    labels = [NODES[i] for i in order]
    x = np.arange(len(order))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    for k, m in enumerate(models):
        e = [rmse(y[:, i], preds[m][:, i]) for i in order]
        ax.bar(x + (k - (len(models) - 1) / 2) * w, e, w, color=COLOR[m], label=LABEL.get(m, m))
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("RMSE (passengers)")
    ax.axvline(n_cp - 0.5, color="0.5", lw=0.8, ls=":")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    savefig(fig, out, f"rmse_by_node_{tag}")


def fig_schedule_source(sources, out, tag, model="TDN+Sched"):
    """sources: ordered dict name -> results json path (clean/published/realized/mixed)."""
    names = list(sources)
    scopes = ["checkpoints", "gates", "overall"]
    vals = np.zeros((len(names), len(scopes)))
    for i, nm in enumerate(names):
        r = json.load(open(sources[nm]))[model]
        vals[i] = [r[s]["RMSE"] for s in scopes]
    x = np.arange(len(scopes))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    shades = ["#d62728", "#ff9896", "#1f77b4", "#2ca02c"]
    for i, nm in enumerate(names):
        ax.bar(x + (i - (len(names) - 1) / 2) * w, vals[i], w, color=shades[i % 4], label=nm)
    ax.set_xticks(x, ["Process nodes", "Gates", "All nodes"])
    ax.set_ylabel(f"RMSE, {LABEL.get(model, model)}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Schedule used as covariate", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    savefig(fig, out, f"schedule_source_{tag}")


def main(preds_path, out, disruption):
    os.makedirs(out, exist_ok=True)
    tag = "15min" if "15min" in preds_path else "1h"
    d = np.load(preds_path, allow_pickle=True)
    y = d["y_true"]
    stamps = d["timestamps"].astype("datetime64[ns]")
    preds = {k[5:]: d[k] for k in d.files if k.startswith("pred_")}
    fig_pred_vs_actual(y, preds, out, tag)
    fig_timeseries(y, preds, stamps, out, tag)
    fig_rmse_by_horizon(y, preds, out, tag)
    fig_rmse_by_node(y, preds, out, tag)
    n = 4
    if disruption:
        fig_schedule_source(disruption, out, tag)
        n += 1
    print(f"{n} figures (pdf+png) -> {out}/  ({tag}; models: {sorted(preds)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default="figs_mco")
    ap.add_argument("--disruption", nargs="*", default=[],
                    help="name=results.json pairs, e.g. clean=... published=... realized=... mixed=...")
    a = ap.parse_args()
    dis = dict(kv.split("=", 1) for kv in a.disruption)
    main(a.preds, a.out, dis)
