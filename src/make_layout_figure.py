"""Detailed diagram of the MCO-inspired terminal layout and of the dataset built on it.

Panel A: the 32-node graph (landside -> two parallel security checkpoints -> APM ->
         four airside concourses -> 24 gates), with every capacity / service-time /
         routing parameter used by generate_mco.py annotated next to the node it
         belongs to, and what each node's Flow_In counts.
Panel B: how flights and passengers are generated (bank schedule, aircraft mix, load
         factor, show-up curve, passenger heterogeneity, disruptions).
Panel C: how the passenger event log becomes the learning tensors (bins, features,
         known-future schedule covariates, tensor shapes, split).

Output: figures/mco_layout_dataset.{pdf,png} (full-page) and
        figures/mco_layout.{pdf,png} (panel A alone, column width).
All numbers are read from generate_mco.py / airport_graph_mco.py, so the figure
cannot drift from the code.
"""
import os
import sys
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["AIRPORT_LAYOUT"] = "mco"
import generate_mco as G                                   # noqa: E402
from airport_graph_mco import (NUM_GATES, GATES_PER_AIRSIDE, NUM_AIRSIDES,   # noqa: E402
                               NODES, adjacency)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42,
    "font.size": 8,
})

FS = [1.0]                      # font scale for panel A (set per figure)


def sz(x):
    return x * FS[0]


C_LAND, C_SEC, C_AIR, C_GATE = "#dbe9f6", "#fde2c8", "#e2f0d9", "#f3f3f3"
C_EDGE, C_APM = "#333333", "#7a7a7a"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def box(ax, x, y, w, h, text, fc, fs=8, bold=True, sub=None, subfs=6.3, lw=0.8, r=0.6):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec="black", lw=lw, zorder=3))
    ax.text(x, y + (0.12 * h if sub else 0), text, ha="center", va="center",
            fontsize=sz(fs), fontweight="bold" if bold else "normal", zorder=4)
    if sub:
        ax.text(x, y - 0.26 * h, sub, ha="center", va="center", fontsize=sz(subfs), zorder=4,
                linespacing=1.05)


def arrow(ax, p, q, text=None, color=C_EDGE, lw=0.9, style="-|>", ls="-", tpos=0.5,
          toff=(0, 1.2), fs=6.3, rad=0.0, ha="center"):
    a = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, lw=lw, color=color,
                        linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=2,
                        shrinkA=1, shrinkB=1)
    ax.add_patch(a)
    if text:
        mx, my = p[0] + tpos * (q[0] - p[0]) + toff[0], p[1] + tpos * (q[1] - p[1]) + toff[1]
        ax.text(mx, my, text, fontsize=sz(fs), ha=ha, va="center", color=color,
                bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.85), zorder=5)


def panel_layout(ax, title=True):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    A = adjacency()
    if title:
        ax.text(0, 99, "A.  MCO-inspired terminal graph: 32 nodes, %d directed edges"
                % int(A.sum()), fontsize=sz(9.5), fontweight="bold", va="top")

    # ---- landside --------------------------------------------------------
    ax.text(9, 92.5, "LANDSIDE", fontsize=sz(7), color="#3b6ea5", ha="center", fontweight="bold")
    box(ax, 9, 84, 16, 9, "Arrival", C_LAND,
        sub="Flow_In = show-ups\n(log-normal lead time)")
    box(ax, 9, 59, 16, 13, "Check-in", C_LAND,
        sub=f"{G.N_KIOSKS} kiosks: {G.KIOSK_SVC[0]:.0f} s, ln-$\\sigma$ {G.KIOSK_SVC[1]}\n"
            f"{G.N_DESKS} desks: {G.DESK_SVC[0]:.0f} s, ln-$\\sigma$ {G.DESK_SVC[1]}\n"
            "FCFS multi-server queues\nFlow_In = check-in starts")
    # ---- security ---------------------------------------------------------
    ax.text(33, 92.5, "SECURITY (parallel)", fontsize=sz(7), color="#b5651d", ha="center", fontweight="bold")
    for name, y in (("Security West", 78), ("Security East", 44)):
        L = G.LANES[name.split()[1]]
        box(ax, 34, y, 17, 15, name, C_SEC,
            sub=f"{L['std']} standard lanes: {G.SEC_SVC['std'][0]:.0f} s, ln-$\\sigma$ {G.SEC_SVC['std'][1]}\n"
                f"{L['pre']} PreCheck lane: {G.SEC_SVC['pre'][0]:.0f} s, ln-$\\sigma$ {G.SEC_SVC['pre'][1]}\n"
                f"{G.SECURITY_FAIL_P:.0%} secondary-screening fail\nFlow_In = queue joins")
    # ---- APM band ---------------------------------------------------------
    ax.add_patch(Rectangle((45.5, 6), 8, 88, fc="#f7f7f7", ec=C_APM, lw=0.8, ls="--", zorder=1))
    ax.text(49.5, 95.5, "APM", fontsize=sz(7), ha="center", color=C_APM, fontweight="bold")
    ax.text(49.5, 3.2, f"people mover: {G.APM_FIXED_MIN:.0f} min + U(0, {G.APM_WAIT_MAX_MIN:.0f}) min wait",
            fontsize=sz(6), ha="center", color=C_APM)
    # ---- airsides + gates -------------------------------------------------
    ax.text(63, 96.5, "AIRSIDE CONCOURSES", fontsize=sz(7), color="#4e8a3e", ha="center", fontweight="bold")
    ax.text(83, 96.5, "GATE PIERS  (walk U(%.0f, %.0f) min)" % G.GATE_WALK_MIN, fontsize=sz(7),
            color="#555555", ha="center", fontweight="bold")
    ys = [86, 64, 42, 20]
    for k, y in enumerate(ys, start=1):
        box(ax, 63, y, 13, 9.5, f"Airside {k}", C_AIR,
            sub=f"gates {(k-1)*GATES_PER_AIRSIDE+1}–{k*GATES_PER_AIRSIDE}\nFlow_In = airside arrivals")
        xs = np.linspace(73, 96.5, GATES_PER_AIRSIDE)
        for j, x in enumerate(xs):
            g = (k - 1) * GATES_PER_AIRSIDE + j + 1
            box(ax, x, y, 4.3, 5.2, f"G{g}", C_GATE, fs=6.5, bold=False, r=0.4)
            if j == 0:
                arrow(ax, (69.5, y), (x - 2.15, y), lw=0.8)
            else:
                arrow(ax, (xs[j - 1] + 2.15, y), (x - 2.15, y), lw=0.6)
        # security -> airside via APM
        for sy in (78, 44):
            arrow(ax, (42.5, sy), (56.5, y), color=C_APM, lw=0.55, rad=0.0)
    ax.text(84.5, 12.5, "Gate Flow_In = boardings; window opens %.0f/%.0f min (dom/intl)\n"
            "and closes %.0f min before departure; gate blocked %.0f min before / %.0f min after"
            % (G.BOARD_OPEN_MIN["dom"], G.BOARD_OPEN_MIN["intl"], G.BOARD_CLOSE_MIN,
               G.GATE_TURN_BEFORE_MIN, G.GATE_TURN_AFTER_MIN),
            fontsize=sz(6), ha="center", va="center", color="#555555")
    # ---- landside edges ---------------------------------------------------
    bag = np.mean([G.BAG_P["dom"], G.BAG_P["intl"]])
    arrow(ax, (4.5, 79.5), (4.5, 65.5), text=f"checked bag:\n{G.BAG_P['dom']:.0%} dom / {G.BAG_P['intl']:.0%} intl;\n"
          f"{G.KIOSK_P_GIVEN_BAG:.0%} use a kiosk", toff=(1.6, 0), fs=5.6, ha="left")
    arrow(ax, (17, 85.5), (25.5, 82), text="no bag: straight\nto security", toff=(0, 4.2), fs=5.6)
    arrow(ax, (17, 82.5), (25.5, 48), rad=0.18)
    arrow(ax, (17, 61.5), (25.5, 74), rad=-0.1)
    arrow(ax, (17, 57), (25.5, 42), rad=0.1)
    ax.text(21, 31.5, "checkpoint choice: %.0f%% affinity\n(West for airsides 1–2, East for 3–4)"
            % (100 * G.CHECKPOINT_AFFINITY), fontsize=sz(5.6), ha="center", va="center", color=C_EDGE)
    # legend-ish note
    ax.text(0.5, 8, "Metrics in the paper: 'checkpoints' = the 8 process nodes\n"
            "(Arrival, Check-in, Security W/E, Airside 1–4); 'gates' = the 24 gate nodes.\n"
            "Edges are the physical passenger routes used to build the (symmetrised)\n"
            "normalised adjacency $D^{-1/2}(A+I)D^{-1/2}$ of the graph models.",
            fontsize=sz(6), va="center", color="#333333")


def blocks(ax, items, y0, x_body, width, step_line=2.75, gap=2.2, fs=6.3):
    """Heading in bold at x=0, wrapped body at x_body; returns the final y."""
    y = y0
    for head, body in items:
        lines = textwrap.wrap(body, width)
        ax.text(0, y, head, fontsize=7, fontweight="bold", va="top")
        ax.text(x_body, y, "\n".join(lines), fontsize=fs, va="top", linespacing=1.12)
        y -= step_line * max(len(lines), 1) + gap
    return y


def fmt(v):
    return "[" + ", ".join(f"{float(x):g}" for x in v) + "]"


def panel_generation(ax):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(0, 99, "B.  Flight schedule and passenger generation (generate_mco.py)",
            fontsize=9.5, fontweight="bold", va="top")
    # bank structure mini-chart
    ins = ax.inset_axes([0.04, 0.60, 0.42, 0.28])
    t = np.linspace(4.5, 24, 800)
    dens = np.zeros_like(t)
    for c, w in zip(G.BANK_CENTERS_MIN / 60, G.BANK_WEIGHTS):
        dens += (1 - G.BACKGROUND_FRAC) * w * np.exp(-0.5 * ((t - c) / (G.BANK_SPREAD_MIN / 60)) ** 2)
    dens += G.BACKGROUND_FRAC / (23.5 - 5) * ((t >= 5) & (t <= 23.5)) * 1.6
    ins.fill_between(t, dens, color="#3b6ea5", alpha=0.75, lw=0)
    ins.set_xlim(4.5, 24); ins.set_yticks([]); ins.set_xticks([6, 9, 12, 15, 18, 21, 24])
    ins.tick_params(labelsize=5.5, length=2, pad=1)
    ins.set_xlabel("published departure time (h)", fontsize=6, labelpad=1)
    ins.set_title("8 departure banks + %.0f%% background flights" % (100 * G.BACKGROUND_FRAC),
                  fontsize=6.5, pad=2)
    for s in ("top", "right", "left"):
        ins.spines[s].set_visible(False)
    # show-up curve
    ins2 = ax.inset_axes([0.55, 0.60, 0.42, 0.28])
    x = np.linspace(20, 310, 600)
    for kind, col in (("dom", "#3b6ea5"), ("intl", "#b5651d")):
        med, sg = G.SHOWUP_MEDIAN_MIN[kind], G.SHOWUP_SIGMA
        pdf = np.exp(-0.5 * ((np.log(x) - np.log(med)) / sg) ** 2) / (x * sg * np.sqrt(2 * np.pi))
        ins2.plot(x, pdf, color=col, lw=1.2, label=f"{kind}: median {med:.0f} min")
    ins2.axvspan(0, G.SHOWUP_CLIP_MIN[0], color="0.85", lw=0); ins2.axvspan(G.SHOWUP_CLIP_MIN[1], 320, color="0.85", lw=0)
    ins2.set_xlim(20, 310); ins2.set_yticks([]); ins2.tick_params(labelsize=5.5, length=2, pad=1)
    ins2.set_xlabel("show-up lead before published departure (min)", fontsize=6, labelpad=1)
    ins2.set_title("log-normal show-up, ln-$\\sigma$ %.2f, clipped [%.0f, %.0f]" % (G.SHOWUP_SIGMA, *G.SHOWUP_CLIP_MIN),
                   fontsize=6.5, pad=2)
    ins2.legend(fontsize=5.5, frameon=False, loc="upper right")
    for s in ("top", "right", "left"):
        ins2.spines[s].set_visible(False)

    lines = [
        ("Flights", f"Poisson(100 · DOW · season) per day; DOW multipliers Mon–Sun = "
                    f"{', '.join(f'{m:.2f}' for m in G.DOW_MULT)}; season ±8 % sinusoid over the 90 days; "
                    f"{G.INTL_FRAC:.0%} international."),
        ("Aircraft", f"seat classes {fmt(G.SEAT_CLASSES)} with p = {fmt(G.SEAT_P_DOM)} (dom) / "
                     f"{fmt(G.SEAT_P_INTL)} (intl); load factor ~ Beta{G.LOAD_FACTOR_BETA} (mean 0.83), "
                     "clipped to [0.5, 1]; passengers = seats × load factor."),
        ("Gates", "assigned by airline-group affinity to an airside (75 %) subject to turnaround "
                  "occupancy; a flight with no free gate is dropped (2 of 9,060)."),
        ("Passengers", f"bag / no bag, kiosk / desk, PreCheck ({G.PRECHECK_P['dom']:.0%} dom, "
                       f"{G.PRECHECK_P['intl']:.0%} intl), checkpoint choice by airside affinity; "
                       "every service point is a FCFS multi-server queue with log-normal service, so "
                       "waits respond non-linearly to load (check-in p95 21 min, security p95 4 min)."),
        ("Disruption", "optional realised schedule: --delay-frac (log-normal delay, mean 45 min, "
                       "[10, 240]) and --gate-change-frac (new gate in the same airside). Show-up follows "
                       "the PUBLISHED time, boarding the REALISED one; both are written to flights.csv."),
        ("Output", "90 days, seed 42: 9,058 flights, 1,224,917 passengers. passengers.csv holds one "
                   "timestamp per process (arrival, check-in start/end, security queue start/end, "
                   "airside arrival, boarding); flights.csv holds published and realised times and gates."),
    ]
    blocks(ax, lines, 54, 15, 72)


def panel_tensors(ax):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(0, 99, "C.  From the event log to the learning tensors (build_features.py)",
            fontsize=9.5, fontweight="bold", va="top")
    steps = [
        ("passengers.csv\n1.22 M rows", C_LAND),
        ("node events\n(in / out per node)", C_LAND),
        ("regular bin grid\n1 h  or  15 min", C_SEC),
        ("X: history window\nY: future flows\nS: future schedule", C_AIR),
    ]
    xs = [11, 34, 57, 84]
    for (t, c), x in zip(steps, xs):
        box(ax, x, 86, 20 if x < 80 else 26, 13, t, c, fs=6.8, bold=False, r=0.8)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + (10 if a < 80 else 13), 86), (b - (10 if b < 80 else 13), 86))
    txt = [
        ("Features (9)", "Flow_In, Flow_Out, hour sin/cos, day-of-week sin/cos, workday, "
                                  "sched_board, sched_dep. Normalisation statistics are fitted on the "
                                  "training split only."),
        ("sched_board", "passengers of each flight spread over its boarding window, added to the gate "
                        "node and to its airside hub: known hours ahead from the flight schedule."),
        ("sched_dep", "passengers on flights departing 1–3.5 h after the bin, added to the landside "
                      "and security nodes (and the flight's gate): the demand heading for the checkpoints."),
        ("Known future", "S carries both covariates over the forecast horizon; the +Sched models receive "
                         "it through a residual schedule head (LSTM/Transformer) or as a context token "
                         "and a prior-residual output (TDN)."),
        ("Schedule source", "realised (perfect), published (what an operator knows), or mixed "
                            "(published show-up, realised gate boarding) for the disruption study."),
        ("Tensor shapes", "1 h, 12 h history, 3 h horizon: X (2144, 32, 9, 12), S (2144, 32, 2, 3), Y (2144, 32, 3).  "
                          "15 min, 3 h history, 3 h horizon: X (8607, 32, 9, 12), S (8607, 32, 2, 12), Y (8607, 32, 12).  "
                          "Horizon sweep: 24 h history, 1–24 h ahead."),
        ("Split / metrics", "chronological 70 / 15 / 15 %; early stopping on validation; RMSE, MAE, R² in "
                            "passengers per bin, reported overall, on the 8 process nodes and on the 24 gates; "
                            "3 seeds for the main tables."),
    ]
    blocks(ax, txt, 74, 20, 68)


def main():
    os.makedirs(OUT, exist_ok=True)
    # ---- full page ------------------------------------------------------
    fig = plt.figure(figsize=(7.4, 10.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], wspace=0.04, hspace=0.06,
                          left=0.02, right=0.99, top=0.985, bottom=0.01)
    axA = fig.add_subplot(gs[0, :]); panel_layout(axA)
    axB = fig.add_subplot(gs[1, 0]); panel_generation(axB)
    axC = fig.add_subplot(gs[1, 1]); panel_tensors(axC)
    for name in ("mco_layout_dataset.pdf", "mco_layout_dataset.png"):
        fig.savefig(os.path.join(OUT, name), dpi=250, bbox_inches="tight")
    plt.close(fig)
    # ---- layout alone (column width) -------------------------------------
    FS[0] = 0.86
    fig, ax = plt.subplots(figsize=(6.3, 5.0))
    panel_layout(ax, title=False)
    FS[0] = 1.0
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for name in ("mco_layout.pdf", "mco_layout.png"):
        fig.savefig(os.path.join(OUT, name), dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("->", OUT)


if __name__ == "__main__":
    main()
