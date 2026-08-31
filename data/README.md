# Data (not tracked)

Raw passenger-flow data is not distributed with this repository due to
operational sensitivity. Do NOT commit data files.

Expected input schema (CSV, one row per node per time step):

| column        | type     | description                              |
|---------------|----------|------------------------------------------|
| timestamp     | datetime | observation time (e.g., 15-min bins)     |
| node_id       | str      | checkpoint/zone identifier               |
| passenger_cnt | int      | passengers processed in the interval     |
| ...           |          | optional covariates (flights, staffing)  |

Plus an edge list `edges.csv`: `src_node_id,dst_node_id,weight`.
