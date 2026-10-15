# NLDC Daily ETL Pipeline

A local, Dockerized, daily batch ETL pipeline. It reads raw **NLDC Excel** files, runs them through a **medallion architecture** (bronze, silver, gold) and publishes a feature-engineered **hourly demand table** as Parquet. **Apache Airflow** runs it once a day and **PySpark** does the processing.

Everything runs locally in Docker, with no cloud services. All timestamps are **IST (Asia/Kolkata)**, with no UTC conversion anywhere.

## What it does

Takes `timestamp` and `demand` from the Excel files and produces:

- `gold/hourly_master`: one row per hour, with `demand_mw` plus calendar, lag, rolling and change features. This is the main output.
- `gold/daily_agg`: one row per day (mean, peak, min, load factor, energy, completeness).

Wind, solar and the other source columns stay in bronze and are not carried forward.

## Architecture

```
Excel ─► LANDING ─► BRONZE ─► SILVER ─► GOLD ─► ARCHIVE
                      │          │
                      └──────────┴──► QUARANTINE (rejected files/rows + reason)
```

| Layer | Content |
|---|---|
| Landing | Raw Excel drops. Never edited. |
| Bronze | Raw data as Parquet, all columns as strings, plus lineage columns. Partitioned by `ingest_date`. |
| Silver | Clean, typed, native grain (5-minute or hourly). Partitioned by `data_month`. |
| Gold | Hourly resample, hourly spine, features. Partitioned by `data_month`. |
| Quarantine | Bad files and rows, each with a reason code. |

More detail is in [docs/architecture.md](docs/architecture.md).

## Source layouts

| | Layout A (SCADA export) | Layout B (Report) |
|---|---|---|
| Data sheet | `Sheet1` | `Report` |
| Grain | 5-minute | Hourly |
| Header rows | 2 | 1 |
| Timestamp | Excel datetime | Text `DD-MM-YYYY HH:MM:SS` |
| Demand column | `NLDC_DEMAND\|P` | `Demand (MW)` |
| Junk | none | 6 footer rows (Minimum, Maximum, Average, Sum ...) |

A file matching neither layout is quarantined as `unknown_layout`. Layouts are defined in `src/config.py`.

## Data rules

- Rejected rows go to `data/quarantine/rows` with a reason: `footer_row`, `unparseable_timestamp`, `non_numeric_demand`, `null_demand`, `demand_nonpositive`, `demand_out_of_range`.
- Gold hourly demand is the **mean of valid readings** in the hour. For 5-minute sources it is null if fewer than `MIN_READINGS_PER_HOUR` (9 of 12) valid readings exist.
- Gold has **24 rows for every day that has any data**. Hours with no data are kept as null rows. Days with no source data get no rows. Demand is never imputed.
- Lag and rolling features are **timestamp-based** and **leakage-safe**: they use only data from `ts_hour - 1h` or earlier. Rolling windows need the full window, otherwise they are null.
- Reruns are **idempotent** (dynamic partition overwrite per `data_month`). Gold reads `LOOKBACK_DAYS` before each month and also rebuilds the following month, so features stay correct on backfills.
- Excel files are archived only after gold validation passes.

## DAG: `nldc_daily_etl` (06:00 IST daily)

```
check_landing ► bronze ► validate_bronze ► silver ► validate_silver ► gold ► validate_gold ► archive_landing
```

A failed validation stops the DAG. Bad data never flows downstream.

## Project structure

```
├── dags/nldc_daily_etl.py     # the DAG
├── src/
│   ├── config.py              # paths, timezone, layouts, thresholds (single source of truth)
│   ├── ingest_bronze.py       # landing Excel -> bronze
│   ├── clean_silver.py        # bronze -> silver (+ quarantine)
│   ├── build_gold.py          # silver -> hourly_master + daily_agg
│   ├── features.py            # calendar, lag, rolling, change features
│   ├── quality.py             # validation gates per layer
│   └── utils.py               # logging, file moves, Spark session
├── tests/                     # pytest suite (bronze, silver, gold)
├── data/                      # mounted volume (git-ignored)
├── Dockerfile
└── docker-compose.yml
```

## Quick start

Requirements: Docker and Docker Compose.

```bash
# 1. run containers as your user so files in ./data are yours
echo "AIRFLOW_UID=$(id -u)" > .env

# 2. build the image and initialise Airflow (creates user admin / admin)
docker compose build
docker compose up airflow-init

# 3. start Postgres, the webserver and the scheduler
docker compose up -d
```

1. Copy your Excel files into `data/landing/`.
2. Open http://localhost:8080 (login `admin` / `admin`).
3. Unpause and trigger `nldc_daily_etl`.
4. Read the results:

```python
import pandas as pd
hourly = pd.read_parquet("data/gold/hourly_master")
daily = pd.read_parquet("data/gold/daily_agg")
```

Stop everything with `docker compose down`.

## Tests

```bash
docker compose run --rm --entrypoint "python -m pytest" airflow-scheduler /opt/airflow/tests -v
```

Tests generate small Excel files on the fly, so no binary fixtures are committed.

## Configuration

Everything tunable is in `src/config.py`: demand valid range, `MIN_READINGS_PER_HOUR`, `LOOKBACK_DAYS`, lag list, rolling windows, `MAX_NULL_DEMAND_RATE`, Spark memory.

## Resources

Tuned for a 16 GB host: Spark in local mode (`local[2]`, 2 GB driver), a single Airflow scheduler and the LocalExecutor.