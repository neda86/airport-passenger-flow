"""MCO-inspired airport departure simulator with realistic distributions.

Run with the matching layout:  AIRPORT_LAYOUT=mco python generate_mco.py --out data

What is more realistic than generate_data.py (the linear-layout generator):

  SCHEDULE   Bank-structured departures (waves), not a smooth Poisson rate;
             day-of-week multipliers (weekends busier, tourism airport) and a
             slow seasonal drift; ~12% international flights.
  AIRCRAFT   Seat-class mix (regional 70 / narrow-body 160 / large NB 190 /
             wide-body 280) x Beta-distributed load factor (mean ~0.83),
             instead of a constant 100 passengers per flight.
  SHOW-UP    Log-normal lead time before departure (median ~100 min domestic,
             ~150 min international, long early tail), instead of a uniform
             window. Show-up is relative to the PUBLISHED departure.
  PASSENGERS Heterogeneous: checked bag or not (bag-less skip check-in),
             kiosk vs counter, PreCheck vs standard security lanes.
  LAYOUT     Two parallel security checkpoints (West/East) chosen by airside
             affinity, APM transit to four airside concourses, 24 gates.
  QUEUES     Every service point is a FCFS multi-server queue with log-normal
             service times, so waits respond nonlinearly to load.
  DISRUPTION Optional --delay-frac / --gate-change-frac produce a REALIZED
             schedule that differs from the PUBLISHED one (both are written
             to flights.csv), for schedule-imperfection experiments. Defaults
             are 0, i.e. published == realized.

Output: passengers.csv + flights.csv (columns documented at the bottom).
"""
import argparse
import heapq
import numpy as np
import pandas as pd

from airport_graph_mco import (NUM_GATES, NUM_AIRSIDES, GATES_PER_AIRSIDE,
                               airside_of_gate)

EPOCH = pd.Timestamp("2024-01-01 00:00:00")
MIN = 60.0
HOUR = 3600.0

# ---------------------------------------------------------------- schedule
BANK_CENTERS_MIN = np.array([6.5, 8.75, 11.0, 13.25, 15.5, 17.75, 20.0, 22.0]) * 60
BANK_WEIGHTS = np.array([0.14, 0.15, 0.13, 0.13, 0.14, 0.14, 0.11, 0.06])
BANK_SPREAD_MIN = 30.0
BACKGROUND_FRAC = 0.15                 # flights not in a bank (uniform 05:00-23:30)
DOW_MULT = np.array([1.00, 0.92, 0.92, 1.00, 1.12, 1.05, 1.10])   # Mon..Sun
INTL_FRAC = 0.12

# ---------------------------------------------------------------- aircraft
SEAT_CLASSES = np.array([70, 160, 190, 280])
SEAT_P_DOM = np.array([0.18, 0.62, 0.13, 0.07])
SEAT_P_INTL = np.array([0.02, 0.35, 0.28, 0.35])
LOAD_FACTOR_BETA = (17.0, 3.5)         # mean ~0.83

# ---------------------------------------------------------------- passengers
SHOWUP_MEDIAN_MIN = {"dom": 100.0, "intl": 150.0}
SHOWUP_SIGMA = 0.38
SHOWUP_CLIP_MIN = (25.0, 300.0)
BAG_P = {"dom": 0.55, "intl": 0.75}
KIOSK_P_GIVEN_BAG = 0.60
PRECHECK_P = {"dom": 0.30, "intl": 0.15}
SECURITY_FAIL_P = 0.03
CHECKPOINT_AFFINITY = 0.75             # P(West | airside 1-2) = P(East | airside 3-4)

# ---------------------------------------------------------------- capacities
N_KIOSKS, KIOSK_SVC = 12, (100.0, 0.40)      # (mean s, lognormal sigma)
N_DESKS, DESK_SVC = 14, (210.0, 0.45)
LANES = {"West": {"std": 4, "pre": 1}, "East": {"std": 4, "pre": 1}}
SEC_SVC = {"std": (32.0, 0.35), "pre": (20.0, 0.30)}
APM_FIXED_MIN, APM_WAIT_MAX_MIN = 5.0, 4.0
GATE_WALK_MIN = (1.0, 5.0)
BOARD_OPEN_MIN = {"dom": 40.0, "intl": 50.0}  # before departure
BOARD_CLOSE_MIN = 15.0
GATE_TURN_BEFORE_MIN, GATE_TURN_AFTER_MIN = 30.0, 40.0


def lognormal_mean(rng, mean, sigma, size):
    """Log-normal samples with the given arithmetic mean and log-sigma."""
    mu = np.log(mean) - 0.5 * sigma ** 2
    return rng.lognormal(mu, sigma, size)


def fcfs_multiserver(arrivals, services, n_servers):
    free_at = [0.0] * n_servers
    heapq.heapify(free_at)
    starts = np.empty_like(arrivals)
    ends = np.empty_like(arrivals)
    for i, (a, s) in enumerate(zip(arrivals, services)):
        t = heapq.heappop(free_at)
        st = max(a, t)
        starts[i], ends[i] = st, st + s
        heapq.heappush(free_at, st + s)
    return starts, ends


def run_queue(df, mask, join_col, n_srv, svc, rng, out_start, out_end):
    """FCFS queue on the rows where mask is True, joined at join_col."""
    idx = df.index[mask]
    if len(idx) == 0:
        return
    order = idx[np.argsort(df.loc[idx, join_col].values, kind="stable")]
    arr = df.loc[order, join_col].values
    st, en = fcfs_multiserver(arr, lognormal_mean(rng, svc[0], svc[1], len(order)), n_srv)
    df.loc[order, out_start] = st
    df.loc[order, out_end] = en


# ---------------------------------------------------------------- flights
def sample_schedule(rng, days, flights_per_day):
    """Published departure times (seconds from EPOCH) with bank structure."""
    times = []
    for d in range(days):
        dow = (EPOCH + pd.Timedelta(days=d)).weekday()
        season = 1.0 + 0.08 * np.sin(2 * np.pi * d / max(days, 1))
        n = rng.poisson(flights_per_day * DOW_MULT[dow] * season)
        n_bg = rng.binomial(n, BACKGROUND_FRAC)
        n_bank = n - n_bg
        banks = rng.choice(len(BANK_CENTERS_MIN), size=n_bank, p=BANK_WEIGHTS)
        t_bank = BANK_CENTERS_MIN[banks] + rng.normal(0, BANK_SPREAD_MIN, n_bank)
        t_bg = rng.uniform(5 * 60, 23.5 * 60, n_bg)
        t = np.concatenate([t_bank, t_bg])
        t = np.clip(t, 5 * 60, 23.9 * 60)
        times.append(d * 24 * 60 + t)
    return np.sort(np.concatenate(times)) * MIN


def assign_gates(rng, deps, board_start, groups):
    """Gate per flight honouring airside affinity and turnaround occupancy."""
    gate_free_at = np.zeros(NUM_GATES + 1)
    gates = np.zeros(len(deps), dtype=int)
    for i, (dep, bs, grp) in enumerate(zip(deps, board_start, groups)):
        use0, use1 = bs - GATE_TURN_BEFORE_MIN * MIN, dep + GATE_TURN_AFTER_MIN * MIN
        pref = grp if rng.uniform() < 0.75 else rng.integers(1, NUM_AIRSIDES + 1)
        pref_gates = [g for g in range(1, NUM_GATES + 1)
                      if airside_of_gate(g) == pref and gate_free_at[g] <= use0]
        cands = pref_gates or [g for g in range(1, NUM_GATES + 1) if gate_free_at[g] <= use0]
        if not cands:
            gates[i] = 0                         # dropped: no gate available
            continue
        g = int(rng.choice(cands))
        gates[i] = g
        gate_free_at[g] = use1
    return gates


def main(out_dir, days, flights_per_day, seed, delay_frac, gate_change_frac,
         lane_scale, lf_mean=None, showup_scale=1.0):
    rng = np.random.default_rng(seed)
    # Sensitivity knobs (defaults reproduce the paper's dataset exactly):
    #   lf_mean       target mean load factor; the Beta concentration (a+b) is
    #                 kept, so only the mean shifts (None = 17/(17+3.5) ~ 0.83)
    #   showup_scale  multiplies the show-up median (1.2 = passengers arrive
    #                 20% earlier, 0.8 = later); sigma and clip unchanged
    lf_beta = LOAD_FACTOR_BETA
    if lf_mean is not None:
        conc = sum(LOAD_FACTOR_BETA)
        lf_beta = (lf_mean * conc, (1.0 - lf_mean) * conc)

    # ---- published schedule ----
    sched_dep = sample_schedule(rng, days, flights_per_day)
    n_fl = len(sched_dep)
    intl = rng.uniform(size=n_fl) < INTL_FRAC
    seats = np.where(intl,
                     rng.choice(SEAT_CLASSES, n_fl, p=SEAT_P_INTL),
                     rng.choice(SEAT_CLASSES, n_fl, p=SEAT_P_DOM))
    lf = np.clip(rng.beta(*lf_beta, n_fl), 0.5, 1.0)
    n_pax = np.maximum(1, np.round(seats * lf)).astype(int)
    groups = rng.integers(1, NUM_AIRSIDES + 1, n_fl)          # airline group -> airside
    kind = np.where(intl, "intl", "dom")
    open_min = np.where(intl, BOARD_OPEN_MIN["intl"], BOARD_OPEN_MIN["dom"])
    sched_bs = sched_dep - open_min * MIN
    sched_be = sched_dep - BOARD_CLOSE_MIN * MIN

    # ---- realized schedule (disruptions; identical to published by default) ----
    delayed = rng.uniform(size=n_fl) < delay_frac
    delay_s = np.where(delayed, np.clip(lognormal_mean(rng, 45 * MIN, 0.6, n_fl), 10 * MIN, 240 * MIN), 0.0)
    dep = sched_dep + delay_s
    bs, be = sched_bs + delay_s, sched_be + delay_s

    sched_gate = assign_gates(rng, sched_dep, sched_bs, groups)
    gate = sched_gate.copy()
    changed = (rng.uniform(size=n_fl) < gate_change_frac) & (sched_gate > 0)
    for i in np.where(changed)[0]:
        same_air = [g for g in range(1, NUM_GATES + 1)
                    if airside_of_gate(g) == airside_of_gate(sched_gate[i]) and g != sched_gate[i]]
        gate[i] = int(rng.choice(same_air))
    keep = gate > 0
    print(f"{n_fl} scheduled flights over {days} days; {(~keep).sum()} dropped (no gate)")

    fl = pd.DataFrame({
        "Flight No": [f"F{i:05d}" for i in range(n_fl)],
        "Airline Group": groups, "International": intl,
        "Seats": seats, "Load Factor": np.round(lf, 3), "Num Passengers": n_pax,
        "Sched Gate": sched_gate, "Gate": gate,
        "sched_dep": sched_dep, "sched_bs": sched_bs, "sched_be": sched_be,
        "dep": dep, "bs": bs, "be": be, "Delay Min": np.round(delay_s / MIN, 1),
        "kind": kind,
    })[keep].reset_index(drop=True)
    fl["Airside"] = [airside_of_gate(g) for g in fl["Gate"]]

    # ---- passengers (vectorized) ----
    rep = np.repeat(fl.index.values, fl["Num Passengers"].values)
    P = len(rep)
    pk = fl["kind"].values[rep]
    is_intl = pk == "intl"
    med = np.where(is_intl, SHOWUP_MEDIAN_MIN["intl"], SHOWUP_MEDIAN_MIN["dom"]) * showup_scale
    showup = np.exp(np.log(med) + rng.normal(0, SHOWUP_SIGMA, P))
    showup = np.clip(showup, *SHOWUP_CLIP_MIN)
    arr_s = fl["sched_dep"].values[rep] - showup * MIN        # relative to PUBLISHED dep

    has_bag = rng.uniform(size=P) < np.where(is_intl, BAG_P["intl"], BAG_P["dom"])
    ci_type = np.where(~has_bag, "none",
                       np.where(rng.uniform(size=P) < KIOSK_P_GIVEN_BAG, "self", "manned"))
    precheck = rng.uniform(size=P) < np.where(is_intl, PRECHECK_P["intl"], PRECHECK_P["dom"])
    airside = fl["Airside"].values[rep]
    west_p = np.where(airside <= 2, CHECKPOINT_AFFINITY, 1 - CHECKPOINT_AFFINITY)
    checkpoint = np.where(rng.uniform(size=P) < west_p, "West", "East")
    passed = rng.uniform(size=P) >= SECURITY_FAIL_P

    pdf = pd.DataFrame({
        "flight": rep, "arr_s": arr_s, "Has Bag": has_bag, "Check-in Type": ci_type,
        "PreCheck": precheck, "Checkpoint": checkpoint, "Airside": airside,
        "passed": passed, "board_lo": fl["bs"].values[rep], "board_hi": fl["be"].values[rep],
    })
    pdf["ci_start"] = np.nan; pdf["ci_end"] = np.nan

    # ---- queues ----
    run_queue(pdf, pdf["Check-in Type"] == "self", "arr_s", N_KIOSKS, KIOSK_SVC, rng, "ci_start", "ci_end")
    run_queue(pdf, pdf["Check-in Type"] == "manned", "arr_s", N_DESKS, DESK_SVC, rng, "ci_start", "ci_end")
    pdf["sec_join"] = pdf["ci_end"].fillna(pdf["arr_s"])      # bag-less go straight to security
    pdf["sec_start"] = np.nan; pdf["sec_end"] = np.nan
    for side in ["West", "East"]:
        for lane, pre in [("std", False), ("pre", True)]:
            m = (pdf["Checkpoint"] == side) & (pdf["PreCheck"] == pre)
            n_lanes = max(1, int(round(LANES[side][lane] * lane_scale)))
            run_queue(pdf, m, "sec_join", n_lanes, SEC_SVC[lane], rng, "sec_start", "sec_end")

    pdf["airside_arr"] = pdf["sec_end"] + (APM_FIXED_MIN + rng.uniform(0, APM_WAIT_MAX_MIN, P)) * MIN
    at_gate = pdf["airside_arr"] + rng.uniform(*GATE_WALK_MIN, P) * MIN
    board = np.where(pdf["passed"].values,
                     np.maximum(at_gate.values, rng.uniform(pdf["board_lo"].values, pdf["board_hi"].values)),
                     np.nan)
    board = np.where(board > pdf["board_hi"].values, np.nan, board)
    pdf["board"] = board

    # ---- diagnostics ----
    ci_w = (pdf["ci_start"] - pdf["arr_s"]).dropna() / MIN
    print(f"{len(fl)} flights, {P} passengers | intl {is_intl.mean():.0%} | bags {has_bag.mean():.0%} "
          f"| PreCheck {precheck.mean():.0%} | West share {(checkpoint=='West').mean():.0%}")
    print(f"show-up lead (min): median {np.median(showup):5.0f}  p10 {np.percentile(showup,10):5.0f}  p90 {np.percentile(showup,90):5.0f}")
    print(f"check-in wait (min): mean {ci_w.mean():5.1f}  p95 {ci_w.quantile(.95):6.1f}  max {ci_w.max():6.1f}")
    for side in ["West", "East"]:
        m = pdf["Checkpoint"] == side
        w = (pdf.loc[m, "sec_start"] - pdf.loc[m, "sec_join"]) / MIN
        print(f"security {side:4s} wait (min): mean {w.mean():5.1f}  p95 {w.quantile(.95):6.1f}  max {w.max():6.1f}")
    missed = np.isnan(board).sum() - (~passed).sum()
    print(f"missed boarding: {missed} ({missed/P:.1%})  failed security: {(~passed).sum()}")
    if delay_frac or gate_change_frac:
        print(f"disruption: {delayed[keep].sum()} delayed flights, {changed[keep].sum()} gate changes")

    # ---- write ----
    ts = lambda s: EPOCH + pd.to_timedelta(s, unit="s")
    out = pd.DataFrame({
        "Passenger ID": [f"{fl['Flight No'].values[r]}_P{i}" for i, r in enumerate(rep)],
        "Flight No": fl["Flight No"].values[rep], "Gate": fl["Gate"].values[rep],
        "Airside": airside, "Checkpoint": checkpoint,
        "Has Bag": has_bag, "PreCheck": precheck, "International": is_intl,
        "Arrival Time": ts(pdf["arr_s"]), "Check-in Type": ci_type,
        "Check-in Start": ts(pdf["ci_start"]), "Check-in End": ts(pdf["ci_end"]),
        "Security Queue Start": ts(pdf["sec_start"]), "Security Queue End": ts(pdf["sec_end"]),
        "Airside Arrival": ts(pdf["airside_arr"]),
        "Boarding Start": ts(pdf["board_lo"]), "Boarding End": ts(pdf["board_hi"]),
        "Boarding Time": ts(pd.Series(board)),
    })
    flights_out = pd.DataFrame({
        "Flight No": fl["Flight No"], "Airline Group": fl["Airline Group"],
        "Airside": fl["Airside"], "Gate": fl["Gate"], "Sched Gate": fl["Sched Gate"],
        "Seats": fl["Seats"], "Load Factor": fl["Load Factor"],
        "Num Passengers": fl["Num Passengers"], "International": fl["International"],
        "Sched Departure": ts(fl["sched_dep"]), "Sched Boarding Start": ts(fl["sched_bs"]),
        "Sched Boarding End": ts(fl["sched_be"]),
        "Departure": ts(fl["dep"]), "Boarding Start": ts(fl["bs"]), "Boarding End": ts(fl["be"]),
        "Delay Min": fl["Delay Min"],
    })
    import os; os.makedirs(out_dir, exist_ok=True)
    out.to_csv(f"{out_dir}/passengers.csv", index=False)
    flights_out.to_csv(f"{out_dir}/flights.csv", index=False)
    print(f"-> {out_dir}/passengers.csv, flights.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--flights-per-day", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--delay-frac", type=float, default=0.0,
                    help="fraction of flights delayed (realized != published)")
    ap.add_argument("--gate-change-frac", type=float, default=0.0)
    ap.add_argument("--lane-scale", type=float, default=1.0,
                    help="multiply security lane counts (e.g. 0.8 = lanes closed)")
    ap.add_argument("--lf-mean", type=float, default=None,
                    help="mean load factor (sensitivity; default 17/20.5 ~ 0.83)")
    ap.add_argument("--showup-scale", type=float, default=1.0,
                    help="multiply the show-up median (sensitivity; 1.0 = paper setting)")
    a = ap.parse_args()
    main(a.out, a.days, a.flights_per_day, a.seed, a.delay_frac, a.gate_change_frac, a.lane_scale,
         a.lf_mean, a.showup_scale)
