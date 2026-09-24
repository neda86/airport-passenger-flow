"""Simulator statistics for the validation table (simulator vs public statistics).

Computes, from a generated dataset (flights.csv + passengers.csv), the
quantities that can be compared with published MCO / US aviation figures:

    departures per day (mean, min, max, by day of week)
    aircraft-size mix (share of flights by seat class) and mean seats
    passengers per day and per departure
    load-factor distribution (mean, sd, 10/50/90 percentiles)
    flight-bank timing (hourly departure share, peak hours)
    show-up profile (minutes before scheduled departure: percentiles)
    checkpoint throughput (passengers per hour per checkpoint: mean, 95th pct, max)
    and the security waiting time (mean, 90th percentile)

Fill the "public" column of the paper's table by hand from BTS T-100 / FAA /
GOAA statistics; this script produces the "simulator" column only.

Usage:
    python simulator_validation.py --data data [--out sim_stats.json]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def pct(a, q):
    return [float(x) for x in np.percentile(a, q)]


def main(a):
    fl = pd.read_csv(os.path.join(a.data, "flights.csv"), parse_dates=["Sched Departure", "Departure"])
    px = pd.read_csv(os.path.join(a.data, "passengers.csv"),
                     parse_dates=["Arrival Time", "Security Queue Start", "Security Queue End", "Boarding Time"])
    fl["day"] = fl["Sched Departure"].dt.date
    fl["hour"] = fl["Sched Departure"].dt.hour
    per_day = fl.groupby("day").size()
    dow = fl.groupby(fl["Sched Departure"].dt.dayofweek).size() / fl["day"].nunique() * 7
    seat_mix = (fl["Seats"].value_counts(normalize=True).sort_index() * 100).round(1)
    # departing passengers per scheduled-departure day, from the flight table (passengers.csv
    # grouped by boarding time drops passengers without a boarding record and shifts post-midnight flights)
    pax_day = fl.groupby("day")["Num Passengers"].sum()
    seat_day = fl.groupby("day")["Seats"].sum()
    showup_min = (fl.set_index("Flight No").loc[px["Flight No"], "Sched Departure"].values
                  - px["Arrival Time"].values) / np.timedelta64(1, "m")
    px["sec_hour"] = px["Security Queue Start"].dt.floor("h")
    thr = px.groupby(["Checkpoint", "sec_hour"]).size()
    wait_min = (px["Security Queue End"] - px["Security Queue Start"]).dt.total_seconds() / 60
    hourly_share = (fl.groupby("hour").size() / len(fl) * 100).round(1)
    hourly_seat_share = (fl.groupby("hour")["Seats"].sum() / fl["Seats"].sum() * 100).round(1)
    dow_seat_share = (fl.groupby(fl["Sched Departure"].dt.dayofweek)["Seats"].sum() / fl["Seats"].sum() * 100).round(1)
    dow_dep_share = (fl.groupby(fl["Sched Departure"].dt.dayofweek).size() / len(fl) * 100).round(1)

    stats = {
        "days": int(fl["day"].nunique()),
        "departures_per_day": {"mean": float(per_day.mean()), "min": int(per_day.min()), "max": int(per_day.max()),
                               "by_dow_Mon..Sun": [round(float(x), 1) for x in dow.values]},
        "seat_mix_percent": {str(k): float(v) for k, v in seat_mix.items()},
        "mean_seats_per_departure": float(fl["Seats"].mean()),
        "international_share_percent": float(fl["International"].mean() * 100),
        "passengers_per_day": {"mean": float(pax_day.mean()), "min": int(pax_day.min()), "max": int(pax_day.max())},
        "seats_per_day": {"mean": float(seat_day.mean()), "min": int(seat_day.min()), "max": int(seat_day.max())},
        "international_share_of_passengers_percent": float(fl.loc[fl["International"], "Num Passengers"].sum() / fl["Num Passengers"].sum() * 100),
        "passengers_per_departure": float(fl["Num Passengers"].mean()),
        "load_factor": {"mean": float(fl["Load Factor"].mean()), "sd": float(fl["Load Factor"].std()),
                        "p10_p50_p90": pct(fl["Load Factor"], [10, 50, 90])},
        "delay_share_percent": float((fl["Delay Min"] > 0).mean() * 100),
        "hourly_departure_share_percent": {int(k): float(v) for k, v in hourly_share.items()},
        "peak_departure_hours": [int(h) for h in hourly_share.sort_values(ascending=False).index[:4]],
        "hourly_seat_share_percent": {int(k): float(v) for k, v in hourly_seat_share.items()},
        "dow_departure_share_percent_Mon..Sun": [float(x) for x in dow_dep_share.values],
        "dow_seat_share_percent_Mon..Sun": [float(x) for x in dow_seat_share.values],
        "showup_minutes_before_departure": {"p10_p50_p90": pct(showup_min, [10, 50, 90]), "mean": float(showup_min.mean())},
        "checkpoint_throughput_pax_per_hour": {
            cp: {"mean": float(thr[cp].mean()), "p95": float(np.percentile(thr[cp], 95)), "max": int(thr[cp].max())}
            for cp in thr.index.get_level_values(0).unique()},
        "security_wait_minutes": {"mean": float(wait_min.mean()), "p90": float(np.percentile(wait_min, 90)),
                                  "max": float(wait_min.max())},
    }
    print(json.dumps(stats, indent=1))
    if a.out:
        json.dump(stats, open(a.out, "w"), indent=2)
        print("saved ->", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=None)
    main(ap.parse_args())
