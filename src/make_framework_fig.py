"""Method figure for the AIAA paper: the factorial model family and the TDN encoder.
Vector PDF, Times-like font, sized for a full text width (6.5 in).
    python make_framework_fig.py --out figures
"""
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.5, "pdf.fonttype": 42,
})


def box(ax, x, y, w, h, text, fc="#f2f2f2", ec="#333333", lw=0.8, fs=8.5, style="round,pad=0.02,rounding_size=0.06"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, lw=0.9, style="-|>", color="#333333", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9, lw=lw, color=color, ls=ls))


def main(out):
    fig, ax = plt.subplots(figsize=(6.5, 2.75))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

    # ---------------- top row: the shared template
    y, h = 30, 9
    box(ax, 1, y, 15, h, "Flow history\n$X_{t-l+1..t}$\n($N$ nodes, $F$ features)", fs=7.6)
    box(ax, 19, y, 15, h, "Spatial operator\nnone | GCN | GAT\n(per input step)", fs=8)
    box(ax, 37, y, 13, h, "Concatenate\nall $N$ node\nembeddings", fc="#e8eef7", fs=8)
    box(ax, 53, y, 19, h, "Temporal encoder\nLSTM | Transformer | TDN", fs=7.6)
    box(ax, 75, y, 10, h, "Linear\nreadout", fs=8)
    box(ax, 88, y, 11, h, "Forecast\n$\\hat{Y}_{t+1..t+p}$\n($N\\times p$)", fc="#e6f2e6", fs=8)
    for x0, x1 in [(16, 19), (34, 37), (50, 53), (72, 75), (85, 88)]:
        arrow(ax, x0, y + h / 2, x1, y + h / 2)
    ax.text(43.5, y + h + 1.3, "dense learned node mixing\n(present even without a graph)", ha="center", va="bottom", fontsize=7.2, color="#1f4e79")

    # schedule branch
    box(ax, 19, 16, 29, 7.5, "Known-future schedule covariates $S_{t+1..t+p}$\n(boarding load, departure pressure; per node)", fc="#fff3e0", fs=7.4)
    box(ax, 53, 16, 19, 7.5, "Schedule head $g_\\theta(S)$\n(non-TDN models)", fc="#fff3e0", fs=7.6)
    arrow(ax, 48, 19.75, 53, 19.75)
    arrow(ax, 72, 19.75, 93.5, 19.75, style="-"); arrow(ax, 93.5, 19.75, 93.5, 30, style="-|>")
    ax.text(82, 20.5, "+", ha="center", va="bottom", fontsize=10)
    # past values also enter X
    arrow(ax, 19, 19.75, 8.5, 19.75, style="-"); arrow(ax, 8.5, 19.75, 8.5, 30, style="-|>")
    ax.text(13.5, 20.6, "past values\njoin $X$", ha="center", va="bottom", fontsize=6.8)

    # ---------------- bottom row: TDN detail
    yb, hb = 2.5, 8.5
    ax.text(1, yb + hb + 1.2, "TDN encoder (schedule-conditioned variant)", ha="left", va="bottom", fontsize=8.5, weight="bold")
    box(ax, 1, yb, 11, hb, "[Q]\nquery token", fc="#e8eef7")
    box(ax, 14, yb, 26, hb, "step tokens $h_1..h_T$\n$+\\,\\psi([t/(T-1),\\,\\log(1+t)])$\ntime-descriptor encoding")
    box(ax, 42, yb, 13, hb, "[C]\ncontext token\nfrom $S$", fc="#fff3e0")
    box(ax, 58, yb, 16, hb, "2-layer Transformer\nencoder; read [Q]")
    box(ax, 77, yb, 22, hb, "$\\hat{Y} = w\\,S^{\\mathrm{board}} + b + \\alpha\\,\\hat{Y}_{\\mathrm{res}}$\nschedule prior + residual", fc="#e6f2e6")
    arrow(ax, 55, yb + hb / 2, 58, yb + hb / 2); arrow(ax, 74, yb + hb / 2, 77, yb + hb / 2)
    ax.text(41, yb - 1.4, "token sequence", ha="center", va="top", fontsize=7, color="#555555")
    ax.plot([1, 55], [yb - 0.6, yb - 0.6], color="#555555", lw=0.6)

    fig.tight_layout(pad=0.2)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "framework_tdn.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, "framework_tdn.png"), dpi=200, bbox_inches="tight")
    print("wrote", os.path.join(out, "framework_tdn.pdf"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="figures")
    main(ap.parse_args().out)
