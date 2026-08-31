"""Publication figures from saved test predictions.

Reads tensors_<freq>_preds.npz (written by train_models.py) and produces:
  1. pred_vs_actual_<freq>.png   — hexbin scatter per model (replaces draft Fig 5/6)
  2. timeseries_<freq>.png       — 4-day actual vs predicted at Security & a gate
  3. rmse_by_horizon_<freq>.png  — error growth over the 12-step forecast horizon
  4. rmse_by_node_<freq>.png     — per-node RMSE, GCN-LSTM vs LSTM vs MA
  5. improvement_heatmap_<freq>.png — where GCN-LSTM beats LSTM (node x horizon)

Usage: make_plots.py --preds tensors_1h_preds.npz --out figs/
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Match the TRB paper's typography: Times-family serif, ~10pt in print.
# Figures are sized at their true printed width (TRB text block ~6.3 in),
# so fonts are NOT shrunk by \includegraphics scaling.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "Times New Roman",
                   "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
})

from airport_graph import NODES, CHECKPOINTS

MODELS = ["GCN-LSTM", "LSTM", "MovingAverage"]
COLORS = {"GCN-LSTM": "#d62728", "LSTM": "#1f77b4", "MovingAverage": "#7f7f7f"}


def rmse(a, b, axis=None):
    return np.sqrt(np.mean((a - b) ** 2, axis=axis))


def fig_pred_vs_actual(y, preds, out, tag):
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5), sharex=True, sharey=True)
    lim = np.percentile(y, 99.8)
    for ax, m in zip(axes, MODELS):
        p = preds[m]
        hb = ax.hexbin(y.ravel(), p.ravel(), gridsize=45, bins="log",
                       cmap="viridis", extent=(0, lim, 0, lim))
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        r2 = 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)
        ax.set_title(m, fontsize=9)
        ax.text(0.05, 0.95, f"RMSE {rmse(y, p):.1f}\n$R^2$ {r2:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(fc="white", ec="0.6", alpha=0.9,
                          boxstyle="round,pad=0.25"))
        ax.set_xlabel("Actual passenger flow")
    axes[0].set_ylabel("Predicted flow")
    fig.colorbar(hb, ax=axes, shrink=0.85, label="log count")
    fig.savefig(f"{out}/pred_vs_actual_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_timeseries(y, preds, stamps, out, tag):
    """First forecast step (t+1) at Security and a busy gate, ~4 days."""
    sec = NODES.index("Security")
    gate_rmse_by_node = y[:, :, 0].sum(axis=0)
    busy_gate = 4 + int(np.argmax(gate_rmse_by_node[4:]))  # busiest gate node
    steps_per_day = 96 if "15min" in tag else 24
    span = slice(0, 4 * steps_per_day)

    fig, axes = plt.subplots(2, 1, figsize=(6.3, 3.6), sharex=True)
    t = stamps[span]
    for ax, node in zip(axes, [sec, busy_gate]):
        ax.plot(t, y[span, node, 0], "k-", lw=1.6, label="Actual")
        for m in MODELS:
            ax.plot(t, preds[m][span, node, 0], lw=1.1, alpha=0.85,
                    color=COLORS[m], label=m)
        ax.set_title(f"{NODES[node]} — 1-step-ahead forecast")
        ax.set_ylabel("Flow In")
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=4, loc="upper right", fontsize=8, framealpha=0.92)
    axes[1].set_xlabel("Time")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{out}/timeseries_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_rmse_by_horizon(y, preds, out, tag):
    H = y.shape[2]
    step_min = 15 if "15min" in tag else 60
    xs = np.arange(1, H + 1) * step_min / 60.0
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    for m in MODELS:
        e = rmse(y, preds[m], axis=(0, 1))
        ax.plot(xs, e, "o-", color=COLORS[m], label=m)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("RMSE (passengers)")
    ax.set_title(f"Error growth over the forecast horizon ({tag})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/rmse_by_horizon_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_rmse_by_node(y, preds, out, tag):
    n_show = len(CHECKPOINTS) + 6  # 4 checkpoints + 6 busiest gates
    totals = y.sum(axis=(0, 2))
    order = list(range(4)) + list(4 + np.argsort(-totals[4:])[:6])
    labels = [NODES[i] for i in order]

    x = np.arange(len(order))
    w = 0.27
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    for k, m in enumerate(MODELS):
        e = [rmse(y[:, i], preds[m][:, i]) for i in order]
        ax.bar(x + (k - 1) * w, e, w, color=COLORS[m], label=m)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("RMSE (passengers)")
    ax.set_title(f"Per-node RMSE — checkpoints + 6 busiest gates ({tag})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/rmse_by_node_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_improvement_heatmap(y, preds, out, tag):
    """% RMSE improvement of GCN-LSTM over LSTM, per node x horizon step."""
    e_g = rmse(y, preds["GCN-LSTM"], axis=0)   # (N, H)
    e_l = rmse(y, preds["LSTM"], axis=0)
    imp = 100 * (e_l - e_g) / np.where(e_l == 0, 1, e_l)
    v = np.nanpercentile(np.abs(imp), 98)
    fig, ax = plt.subplots(figsize=(5.5, 4.3))
    im = ax.imshow(imp, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_yticks(range(len(NODES)), NODES, fontsize=7)
    ax.set_xticks(range(y.shape[2]), [f"t+{i+1}" for i in range(y.shape[2])],
                  fontsize=7)
    ax.set_xlabel("Forecast step")
    ax.set_title(f"GCN-LSTM RMSE improvement over LSTM, % (red = GCN better) — {tag}")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(f"{out}/improvement_heatmap_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(preds_path, out):
    os.makedirs(out, exist_ok=True)
    tag = "15min" if "15min" in preds_path else "1h"
    d = np.load(preds_path, allow_pickle=True)
    y = d["y_true"]
    stamps = d["timestamps"].astype("datetime64[ns]")
    preds = {m: d[f"pred_{m}"] for m in MODELS}

    fig_pred_vs_actual(y, preds, out, tag)
    fig_timeseries(y, preds, stamps, out, tag)
    fig_rmse_by_horizon(y, preds, out, tag)
    fig_rmse_by_node(y, preds, out, tag)
    fig_improvement_heatmap(y, preds, out, tag)
    print(f"5 figures -> {out}/  ({tag})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default="figs")
    args = ap.parse_args()
    main(args.preds, args.out)
