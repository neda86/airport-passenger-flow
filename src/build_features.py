"""Feature engineering: passenger events -> (samples, nodes, features, seq) tensors.

Fixes vs. the original notebooks (03. Feature Engineering / Airport_FE_p85_LSTM):
  1. Node axis uses the canonical order from airport_graph.NODES — identical to
     the adjacency matrix rows. (Old code: Node_ID from first-appearance order,
     unrelated to the adjacency's declared order => GCN mixed wrong neighbors.)
  2. NO normalization here. Scaling params are fit on the TRAIN SPLIT ONLY
     inside train_models.py. (Old code z-scored then min-maxed the full dataset
     before splitting => leakage, and double-scaled the inputs.)
  3. The time grid is a complete regular grid over the simulation span, so
     sliding windows are always contiguous in real time.
  4. Peak-hour/peak-day flags removed: they were computed from full-dataset
     statistics (leakage). Cyclical time encodings carry the same signal.

Schedule conditioning: if --flights is given, two KNOWN-FUTURE covariates are
built from the published flight schedule (both are known hours in advance in
real operations, so using them over the forecast horizon is NOT leakage):
  sched_board — expected passengers boarding at the node in the bin
                (per gate; Boarding node = sum over gates)
  sched_dep   — passengers on flights departing 1-3.5h after the bin
                (landside "pressure" heading for check-in/security)
They are appended to the history features AND exported as a future tensor
S (samples, nodes, 2, tar_seq) aligned with the forecast horizon.

Usage: build_features.py --freq 15min|1h [--flights flights.csv]
Output: tensors_<freq>.npz with X (samples, nodes, features, in_seq),
        Y (samples, nodes, tar_seq), S (if flights given), timestamps.
"""
import argparse
import numpy as np
import pandas as pd

from airport_graph import NODES, NODE_INDEX, NUM_NODES, CHECKPOINTS

FEATURES = ["Flow_In", "Flow_Out", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "workday"]
SCHED_FEATURES = ["sched_board", "sched_dep"]
TARGET = "Flow_In"


def node_flow_events(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (node, flow-type) event, timestamped."""
    for col in ["Arrival Time", "Check-in Start", "Check-in End",
                "Security Queue Start", "Security Queue End",
                "Boarding Start", "Boarding End", "Boarding Time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    recs = []
    def add(node, start_col, end_col):
        sub = df[[start_col, end_col]].dropna()
        recs.append(pd.DataFrame({"Node": node, "Time": sub[start_col], "Type": "in"}))
        recs.append(pd.DataFrame({"Node": node, "Time": sub[end_col], "Type": "out"}))

    add("Arrival", "Arrival Time", "Check-in Start")
    add("Check-in", "Check-in Start", "Check-in End")
    add("Security", "Security Queue Start", "Security Queue End")
    add("Boarding", "Security Queue End", "Boarding Time")

    gate = df[["Boarding Time", "Gate", "Departure"]] if "Departure" in df else df[["Boarding Time", "Gate"]]
    gate = gate.dropna(subset=["Boarding Time"]).copy()
    gate["Node"] = "Gate " + gate["Gate"].astype(int).astype(str)
    recs.append(pd.DataFrame({"Node": gate["Node"], "Time": gate["Boarding Time"], "Type": "in"}))

    return pd.concat(recs, ignore_index=True)


def schedule_features(flights_csv: str, bins: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """(NUM_NODES, T, 2) known-future schedule covariates on the bin grid."""
    fl = pd.read_csv(flights_csv, parse_dates=["Boarding Start", "Boarding End", "Departure"])
    step = pd.Timedelta(freq)
    t0 = bins[0]
    T = len(bins)
    sched = np.zeros((NUM_NODES, T, 2), dtype=np.float32)
    board_node = NODE_INDEX["Boarding"]
    cp_idx = [NODE_INDEX[c] for c in CHECKPOINTS if c != "Boarding"]

    for _, f in fl.iterrows():
        gate_node = NODE_INDEX[f"Gate {int(f['Gate'])}"]
        pax = float(f["Num Passengers"])

        # sched_board: pax spread over the boarding window, proportional to overlap.
        b0, b1 = f["Boarding Start"], f["Boarding End"]
        dur = (b1 - b0).total_seconds()
        i0 = max(0, int((b0 - t0) / step))
        i1 = min(T - 1, int((b1 - t0) / step))
        for i in range(i0, i1 + 1):
            lo, hi = bins[i], bins[i] + step
            ov = (min(hi, b1) - max(lo, b0)).total_seconds()
            if ov > 0 and dur > 0:
                w = pax * ov / dur
                sched[gate_node, i, 0] += w
                sched[board_node, i, 0] += w

        # sched_dep: landside pressure — flight contributes to bins 1-3.5h
        # before its departure (the arrival window of its passengers).
        d0, d1 = f["Departure"] - pd.Timedelta(hours=3.5), f["Departure"] - pd.Timedelta(hours=1.0)
        j0 = max(0, int((d0 - t0) / step))
        j1 = min(T - 1, int((d1 - t0) / step))
        if j1 >= j0:
            for n in cp_idx:
                sched[n, j0:j1 + 1, 1] += pax
            sched[gate_node, j0:j1 + 1, 1] += pax

    return sched


def build(passengers_csv: str, freq: str, in_seq: int, tar_seq: int,
          flights_csv: str | None = None):
    df = pd.read_csv(passengers_csv)
    events = node_flow_events(df)
    events["Bin"] = events["Time"].dt.floor(freq)

    counts = (events.groupby(["Node", "Bin", "Type"]).size()
              .unstack(fill_value=0).reset_index()
              .rename(columns={"in": "Flow_In", "out": "Flow_Out"}))
    for c in ("Flow_In", "Flow_Out"):
        if c not in counts:
            counts[c] = 0

    # Complete regular grid: every node at every bin over the full span.
    bins = pd.date_range(events["Bin"].min(), events["Bin"].max(), freq=freq)
    grid = pd.MultiIndex.from_product([NODES, bins], names=["Node", "Bin"])
    counts = (counts.set_index(["Node", "Bin"]).reindex(grid)
              .fillna(0).reset_index())

    # Calendar features (deterministic — no leakage possible).
    counts["hour_sin"] = np.sin(2 * np.pi * counts["Bin"].dt.hour / 24)
    counts["hour_cos"] = np.cos(2 * np.pi * counts["Bin"].dt.hour / 24)
    counts["dow_sin"] = np.sin(2 * np.pi * counts["Bin"].dt.weekday / 7)
    counts["dow_cos"] = np.cos(2 * np.pi * counts["Bin"].dt.weekday / 7)
    counts["workday"] = (counts["Bin"].dt.weekday < 5).astype(float)

    # -> arrays indexed [node, time] in CANONICAL node order.
    counts["node_idx"] = counts["Node"].map(NODE_INDEX)
    counts = counts.sort_values(["node_idx", "Bin"])
    T = len(bins)
    feat = np.zeros((NUM_NODES, T, len(FEATURES)), dtype=np.float32)
    for f_i, f in enumerate(FEATURES):
        feat[:, :, f_i] = counts[f].to_numpy().reshape(NUM_NODES, T)
    target = counts[TARGET].to_numpy().reshape(NUM_NODES, T).astype(np.float32)

    sched = None
    if flights_csv is not None:
        sched = schedule_features(flights_csv, bins, freq)
        feat = np.concatenate([feat, sched], axis=2)  # history side

    # Sliding windows.
    n_feat = feat.shape[2]
    n_samples = T - in_seq - tar_seq + 1
    X = np.zeros((n_samples, NUM_NODES, n_feat, in_seq), dtype=np.float32)
    Y = np.zeros((n_samples, NUM_NODES, tar_seq), dtype=np.float32)
    S = (np.zeros((n_samples, NUM_NODES, 2, tar_seq), dtype=np.float32)
         if sched is not None else None)
    for s in range(n_samples):
        X[s] = feat[:, s:s + in_seq, :].transpose(0, 2, 1)
        Y[s] = target[:, s + in_seq:s + in_seq + tar_seq]
        if S is not None:
            S[s] = sched[:, s + in_seq:s + in_seq + tar_seq, :].transpose(0, 2, 1)
    stamps = bins[in_seq: in_seq + n_samples]  # forecast-origin timestamps

    return X, Y, S, stamps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--passengers", default="passengers.csv")
    ap.add_argument("--freq", default="15min", choices=["15min", "1h"])
    ap.add_argument("--in-seq", type=int, default=12,
                    help="history window length in steps")
    ap.add_argument("--tar-seq", type=int, default=12,
                    help="forecast horizon in steps")
    ap.add_argument("--flights", default=None,
                    help="flights.csv for known-future schedule covariates")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    X, Y, S, stamps = build(args.passengers, args.freq, args.in_seq,
                            args.tar_seq, args.flights)
    out = args.out or f"tensors_{args.freq}.npz"
    feats = FEATURES + (SCHED_FEATURES if S is not None else [])
    arrays = dict(X=X, Y=Y,
                  timestamps=np.array(stamps, dtype="datetime64[ns]"),
                  features=np.array(feats), nodes=np.array(NODES))
    if S is not None:
        arrays["S"] = S
    np.savez_compressed(out, **arrays)
    print(f"X {X.shape}  Y {Y.shape}  S {None if S is None else S.shape}  -> {out}")
    print(f"target: mean={Y.mean():.2f} std={Y.std():.2f} "
          f"zeros={np.mean(Y == 0) * 100:.1f}% max={Y.max():.0f}")
