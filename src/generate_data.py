"""Synthetic airport passenger data generator (fixed).

Fixes vs. the original notebook (01. Poisson_... / Airport_Passenger_generator):
  1. True Poisson-process arrivals: passenger count ~ Poisson(lambda), arrival
     instants are uniform order statistics inside the window — which IS a
     homogeneous Poisson process conditioned on the count. The old code pinned
     the first arrival to the exact window start and the last to the exact end
     of an identical 1-hour window for every flight (deterministic artifacts).
  2. Arrival window widened and randomized per flight (early birds vs late
     arrivals), instead of one rigid [T-3h, T-2h] hour for everyone.
  3. All 20 gates are assignable (old code: range(1, 20) => Gate 20 never used,
     silently shrinking the graph to 23 nodes).
  4. Diurnal flight-rate profile + weekend uplift, so the series has real
     daily/weekly seasonality for the model to learn. The old constant
     5 flights/hour, 24h/day gave the models almost no temporal structure.
  5. Deterministic seed for reproducibility.
  6. CAPACITY-CONSTRAINED QUEUES: check-in (self-service kiosks + manned
     desks) and security (parallel lanes) are FCFS multi-server queues, so
     waiting times respond nonlinearly to load. The old code drew service
     durations independently of crowding — no congestion dynamics at all,
     which undercut the whole congestion-forecasting premise.

Output: passengers.csv + flights.csv (same columns the feature-engineering
step expects).
"""
import heapq
import argparse
import numpy as np
import pandas as pd

from airport_graph import NUM_GATES

SEED = 42
EPOCH = pd.Timestamp("2024-01-01 00:00:00")  # simulation t=0

# --- simulation parameters (seconds) ---
SIM_DAYS = 90
BASE_FLIGHTS_PER_HOUR = 5.0
BOARDING_DURATION = (45 * 60, 60 * 60)
DEPART_AFTER_BOARDING = (10 * 60, 30 * 60)
MEAN_PASSENGERS = 100

# Capacity-constrained service points (FCFS multi-server queues).
# Sized so average utilization is moderate but peak-hour bunching produces
# real queues (peak arrivals ~800-900 pax/h).
N_KIOSKS = 15          # self-service check-in
KIOSK_SERVICE = (60, 180)          # 1-3 min
N_DESKS = 26           # manned check-in desks
DESK_SERVICE = (120, 360)          # 2-6 min
N_SECURITY_LANES = 7
SECURITY_SERVICE = (20, 50)        # per-passenger scan time per lane

# Hourly multiplier on the flight rate: quiet overnight, morning & evening peaks.
DIURNAL = np.array([
    0.10, 0.05, 0.05, 0.10, 0.30, 0.70,   # 00-05
    1.20, 1.60, 1.50, 1.20, 1.00, 1.00,   # 06-11
    1.10, 1.20, 1.10, 1.30, 1.50, 1.60,   # 12-17
    1.40, 1.10, 0.80, 0.50, 0.30, 0.15,   # 18-23
])


def sample_flight_times(rng: np.random.Generator, sim_seconds: int) -> np.ndarray:
    """Departure times from an inhomogeneous Poisson process via thinning."""
    lam_max = BASE_FLIGHTS_PER_HOUR * DIURNAL.max() * 1.25 / 3600.0  # weekend headroom
    t, out = 0.0, []
    while True:
        t += rng.exponential(1.0 / lam_max)
        if t >= sim_seconds:
            break
        ts = EPOCH + pd.Timedelta(seconds=t)
        rate = BASE_FLIGHTS_PER_HOUR * DIURNAL[ts.hour] / 3600.0
        if ts.weekday() >= 5:
            rate *= 1.175
        if rng.uniform() < rate / lam_max:
            out.append(t)
    return np.asarray(out)


def poisson_process_arrivals(rng, n, start, end):
    """n arrival instants of a homogeneous Poisson process on [start, end]."""
    return np.sort(rng.uniform(start, end, size=n))


def fcfs_multiserver(arrivals, services, n_servers):
    """FCFS multi-server queue. arrivals must be sorted ascending.
    Returns (service_start, service_end) arrays."""
    free_at = [0.0] * n_servers
    heapq.heapify(free_at)
    starts = np.empty_like(arrivals)
    ends = np.empty_like(arrivals)
    for i, (a, s) in enumerate(zip(arrivals, services)):
        t = heapq.heappop(free_at)
        start = max(a, t)
        end = start + s
        starts[i], ends[i] = start, end
        heapq.heappush(free_at, end)
    return starts, ends


def main(out_dir: str):
    rng = np.random.default_rng(SEED)
    sim_seconds = SIM_DAYS * 24 * 3600

    departures = sample_flight_times(rng, sim_seconds)
    print(f"{len(departures)} flights over {SIM_DAYS} days")

    flights, passengers = [], []
    gate_free_at = np.zeros(NUM_GATES + 1)  # gate_free_at[g] = time gate g frees up

    for k, dep in enumerate(departures):
        boarding_dur = rng.uniform(*BOARDING_DURATION)
        gap = rng.uniform(*DEPART_AFTER_BOARDING)
        boarding_start = dep - gap - boarding_dur
        boarding_end = dep - gap

        # Gate: any gate free from (boarding_start - 1h) onwards; ALL 20 usable.
        usage_start, usage_end = boarding_start - 3600, dep + 3600
        candidates = [g for g in range(1, NUM_GATES + 1) if gate_free_at[g] <= usage_start]
        if not candidates:
            continue  # no free gate; drop the flight
        gate = int(rng.choice(candidates))
        gate_free_at[gate] = usage_end

        n_pax = rng.poisson(MEAN_PASSENGERS)
        dep_ts = EPOCH + pd.Timedelta(seconds=dep)
        if dep_ts.weekday() >= 5:
            n_pax = int(n_pax * rng.uniform(1.15, 1.20))
        if n_pax == 0:
            continue

        flight_no = f"F{k:05d}"
        flights.append({
            "Flight No": flight_no, "Terminal": 1, "Gate": gate,
            "Boarding Start": EPOCH + pd.Timedelta(seconds=boarding_start),
            "Boarding End": EPOCH + pd.Timedelta(seconds=boarding_end),
            "Departure": dep_ts, "Num Passengers": n_pax,
        })

        # Passenger arrival window: randomized per flight, 1.5-3.5h pre-departure.
        w_start = dep - rng.uniform(3.0, 3.5) * 3600
        w_end = dep - rng.uniform(1.5, 2.0) * 3600
        arrivals = poisson_process_arrivals(rng, n_pax, w_start, w_end)
        checkin_type = rng.choice(["self", "manned"], size=n_pax)
        passed = rng.uniform(size=n_pax) < 0.95

        for i in range(n_pax):
            passengers.append({
                "Passenger ID": f"{flight_no}_P{i}", "Flight No": flight_no,
                "Gate": gate, "arr_s": arrivals[i],
                "Check-in Type": checkin_type[i], "passed": passed[i],
                "board_lo": boarding_start, "board_hi": boarding_end,
            })

    pdf = pd.DataFrame(passengers)

    # ---- Phase 2: capacity-constrained queues, shared airport-wide ----
    # Check-in: two FCFS server pools (kiosks / desks), processed in arrival order.
    pdf["ci_start"], pdf["ci_end"] = 0.0, 0.0
    for ctype, n_srv, svc in [("self", N_KIOSKS, KIOSK_SERVICE),
                              ("manned", N_DESKS, DESK_SERVICE)]:
        idx = pdf.index[pdf["Check-in Type"] == ctype]
        order = idx[np.argsort(pdf.loc[idx, "arr_s"].values)]
        arr = pdf.loc[order, "arr_s"].values
        svc_t = rng.uniform(*svc, size=len(order))
        s, e = fcfs_multiserver(arr, svc_t, n_srv)
        pdf.loc[order, "ci_start"] = s   # service start (queue wait = start - arrival)
        pdf.loc[order, "ci_end"] = e

    # Security: one FCFS pool of lanes, processed in order of check-in completion.
    order = pdf.index[np.argsort(pdf["ci_end"].values)]
    arr = pdf.loc[order, "ci_end"].values
    svc_t = rng.uniform(*SECURITY_SERVICE, size=len(order))
    s, e = fcfs_multiserver(arr, svc_t, N_SECURITY_LANES)
    pdf.loc[order, "sec_start"] = s
    pdf.loc[order, "sec_end"] = e

    # Boarding.
    board = np.where(
        pdf["passed"].values,
        np.maximum(pdf["sec_end"].values,
                   rng.uniform(pdf["board_lo"].values, pdf["board_hi"].values)),
        np.nan,
    )
    # Missed flight: cleared security after boarding window closed.
    board = np.where(board > pdf["board_hi"].values, np.nan, board)

    # Diagnostics: does congestion actually happen?
    ci_wait = (pdf["ci_start"] - pdf["arr_s"]) / 60
    sec_wait = (pdf["sec_start"] - pdf["ci_end"]) / 60
    print(f"check-in wait  (min): mean {ci_wait.mean():5.1f}  p95 {ci_wait.quantile(.95):6.1f}  max {ci_wait.max():6.1f}")
    print(f"security wait  (min): mean {sec_wait.mean():5.1f}  p95 {sec_wait.quantile(.95):6.1f}  max {sec_wait.max():6.1f}")
    print(f"missed flights: {np.isnan(board).sum() - (~pdf['passed']).sum()} "
          f"(+ {(~pdf['passed']).sum()} failed security)")

    to_ts = lambda col: EPOCH + pd.to_timedelta(col, unit="s")
    out = pd.DataFrame({
        "Passenger ID": pdf["Passenger ID"], "Flight No": pdf["Flight No"],
        "Gate": pdf["Gate"],
        "Arrival Time": to_ts(pdf["arr_s"]),
        "Check-in Type": pdf["Check-in Type"],
        "Check-in Start": to_ts(pdf["ci_start"]),
        "Check-in End": to_ts(pdf["ci_end"]),
        "Security Queue Start": to_ts(pdf["sec_start"]),
        "Security Queue End": to_ts(pdf["sec_end"]),
        "Boarding Start": to_ts(pdf["board_lo"]),
        "Boarding End": to_ts(pdf["board_hi"]),
        "Boarding Time": to_ts(pd.Series(board)),
    })
    fdf, pdf = pd.DataFrame(flights), out
    fdf.to_csv(f"{out_dir}/flights.csv", index=False)
    pdf.to_csv(f"{out_dir}/passengers.csv", index=False)
    print(f"{len(fdf)} flights, {len(pdf)} passengers -> {out_dir}/")
    print("Gates used:", sorted(fdf["Gate"].unique()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".", help="output directory")
    # --- distribution-shift / experiment knobs (defaults reproduce the
    # baseline dataset exactly; override any subset for a shifted config) ---
    ap.add_argument("--days", type=int, default=SIM_DAYS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--flight-scale", type=float, default=1.0,
                    help="multiplier on BASE_FLIGHTS_PER_HOUR "
                    "(e.g. 1.3 = 30%% denser schedule)")
    ap.add_argument("--kiosks", type=int, default=N_KIOSKS)
    ap.add_argument("--desks", type=int, default=N_DESKS)
    ap.add_argument("--security-lanes", type=int, default=N_SECURITY_LANES)
    args = ap.parse_args()

    SIM_DAYS = args.days
    SEED = args.seed
    BASE_FLIGHTS_PER_HOUR = BASE_FLIGHTS_PER_HOUR * args.flight_scale
    N_KIOSKS, N_DESKS = args.kiosks, args.desks
    N_SECURITY_LANES = args.security_lanes
    if (args.days, args.seed, args.flight_scale, args.kiosks, args.desks,
            args.security_lanes) != (90, 42, 1.0, 15, 26, 7):
        print(f"[shifted config] days={SIM_DAYS} seed={SEED} "
              f"flights/h={BASE_FLIGHTS_PER_HOUR:.2f} kiosks={N_KIOSKS} "
              f"desks={N_DESKS} lanes={N_SECURITY_LANES}")
    main(args.out)
