# data/

Nothing in this directory is tracked. The paper uses simulated data only, and every
dataset is regenerated from the code with a fixed seed:

```
cd src
AIRPORT_LAYOUT=mco python generate_mco.py --out ../data --days 90 --seed 42
```

The generator writes two files.

`flights.csv`, one row per departing flight: flight number, airline group, airside and
gate (scheduled and realized), seats, load factor, number of passengers, international
flag, scheduled and realized departure, boarding start and boarding end, and the delay
in minutes. On the clean dataset the scheduled and realized columns coincide; the
disrupted dataset (`--delay-frac`, `--gate-change-frac`) is where they differ, and
`build_features.py --schedule published|realized|mixed` chooses which of them feed the
covariates.

`passengers.csv`, one row per passenger: flight, gate, airside, checkpoint (West or
East), checked bag, PreCheck and international flags, and the time of every event on
the way through the terminal: arrival at the terminal, check-in (type, start, end),
security queue start and end, arrival at the airside concourse, the boarding window
and the boarding time.

`build_features.py` turns the event log into the tensors the models train on and
saves them as a compressed `.npz` with

| array | shape | content |
|---|---|---|
| `X` | (origins, nodes, features, input steps) | node inflow and outflow, calendar features and the two schedule covariates over the input window |
| `Y` | (origins, nodes, horizon) | node inflows to forecast |
| `S` | (origins, nodes, 2, horizon) | the two known-future covariates over the forecast horizon: scheduled boarding load and departure demand |

together with the feature names, the timestamps of the forecast origins and the node
list. The node order is the one defined in `src/airport_graph_mco.py` (or
`airport_graph_linear.py`), which is also the order of the adjacency matrix used by the
graph operators.

The 90-day baseline dataset is about 370 MB (`passengers.csv`) and takes a minute or
two to generate.
