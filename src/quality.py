"""Validation gates per layer (plan sections 7 and 8: fail loudly, bad data never flows downstream).

Each validate_* function runs every check, collects all failures, and raises one
QualityError listing them, so a single run shows everything that is wrong.
Data is small, so checks read Parquet with pandas/pyarrow instead of starting Spark.
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import config, features
from src.utils import get_logger

log = get_logger(__name__)

_EPS = 1e-9


class QualityError(Exception):
    """A validation gate failed. The DAG treats this as a hard failure (no retry)."""


class _Checks:
    def __init__(self, layer: str):
        self.layer = layer
        self.failures = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.failures.append(message)

    def finish(self) -> None:
        if self.failures:
            for f in self.failures:
                log.error("validate_%s FAILED | %s", self.layer, f)
            raise QualityError(f"validate_{self.layer}: {len(self.failures)} check(s) failed: {self.failures}")
        log.info("validate_%s passed", self.layer)


# --------------------------------------------------------------------------- #
# Bronze
# --------------------------------------------------------------------------- #
def validate_bronze(summary: dict) -> None:
    """Row count equals the Excel data-row count (footer rows included); expected columns present."""
    c = _Checks("bronze")
    files = summary.get("ingested", [])
    c.require(len(files) > 0, "no files were ingested")

    for f in files:
        name, layout = f["file"], config.SOURCE_LAYOUTS[f["layout"]]
        parquet = pq.ParquetFile(f["parquet"])
        on_disk = parquet.metadata.num_rows
        c.require(
            on_disk == f["expected_rows"],
            f"{name}: {on_disk} rows in bronze Parquet but {f['expected_rows']} data rows in Excel",
        )
        expected_cols = {layout["timestamp_col"], layout["demand_col"], *config.BRONZE_META_COLUMNS}
        missing = expected_cols - set(parquet.schema_arrow.names)
        c.require(not missing, f"{name}: missing columns {sorted(missing)}")
    c.finish()


# --------------------------------------------------------------------------- #
# Silver
# --------------------------------------------------------------------------- #
def validate_silver(summary: dict) -> None:
    """Schema matches spec, no duplicate timestamps, demand non-null and in range,
    rejected + kept rows equals bronze rows."""
    c = _Checks("silver")

    # every bronze row of the batch is either kept or quarantined
    rejected = sum(summary["rejected_by_reason"].values())
    kept = summary["silver_rows_from_batch"]
    c.require(
        kept + rejected == summary["bronze_rows"],
        f"row accounting: kept {kept} + rejected {rejected} != bronze {summary['bronze_rows']}",
    )

    months = summary["months_touched"]
    if not months:
        c.require(False, "batch produced no valid silver rows")
        c.finish()
        return

    df = pd.read_parquet(config.SILVER_DIR, filters=[("data_month", "in", months)])
    c.require(
        list(df.columns) == config.SILVER_COLUMNS,
        f"columns {list(df.columns)} != spec {config.SILVER_COLUMNS}",
    )
    c.require(
        len(df) == summary["silver_rows_total_in_touched_months"],
        f"silver has {len(df)} rows in touched months, expected {summary['silver_rows_total_in_touched_months']}",
    )
    c.require(int(df["ts"].isna().sum()) == 0, "null ts found")
    c.require(int(df["ts"].duplicated().sum()) == 0, f"{int(df['ts'].duplicated().sum())} duplicate timestamps")
    c.require(int(df["demand_mw"].isna().sum()) == 0, "null demand_mw found")
    c.require(int((df["demand_mw"] <= 0).sum()) == 0, "non-positive demand_mw found")
    out_of_range = ~df["demand_mw"].between(config.DEMAND_VALID_MIN_MW, config.DEMAND_VALID_MAX_MW)
    c.require(int(out_of_range.sum()) == 0, f"{int(out_of_range.sum())} demand_mw values outside configured range")
    bad_grain = ~df["source_grain"].isin(list(config.EXPECTED_READINGS_PER_HOUR))
    c.require(int(bad_grain.sum()) == 0, "unknown source_grain values")
    c.finish()


# --------------------------------------------------------------------------- #
# Gold
# --------------------------------------------------------------------------- #
def validate_gold(summary: dict = None) -> None:
    """24 rows per covered day, ts_hour unique and contiguous within each day, null rate
    under threshold, feature sanity (cyclical range, calendar range, lags consistent
    with the source hour), plus daily_agg consistency."""
    c = _Checks("gold")

    full = pd.read_parquet(config.HOURLY_MASTER_DIR)          # all gold months (lag source)
    full["data_month"] = full["data_month"].astype(str)
    months = (summary or {}).get("months_rebuilt") or sorted(full["data_month"].unique())
    df = full[full["data_month"].isin(months)].copy()
    c.require(len(df) > 0, f"no hourly_master rows for months {months}")

    # schema
    c.require(
        list(full.columns) == features.feature_columns(),
        "hourly_master columns differ from the spec (see features.feature_columns())",
    )

    # one row per hour, 24 per covered day, contiguous
    c.require(bool(full["ts_hour"].is_unique), "ts_hour is not unique")
    day = df["ts_hour"].dt.date
    per_day = df.groupby(day).agg(rows=("ts_hour", "size"), hours=("ts_hour", lambda s: s.dt.hour.nunique()))
    bad_days = per_day[(per_day["rows"] != 24) | (per_day["hours"] != 24)]
    c.require(bad_days.empty, f"{len(bad_days)} day(s) without exactly 24 distinct hours, e.g. {list(bad_days.index[:3])}")

    # null rate of demand_mw, checked per month so one empty month cannot hide in an average
    null_rate = df["demand_mw"].isna().groupby(df["data_month"]).mean()
    for month, rate in null_rate.items():
        c.require(
            rate <= config.MAX_NULL_DEMAND_RATE,
            f"{month}: demand_mw null rate {rate:.1%} exceeds {config.MAX_NULL_DEMAND_RATE:.0%}",
        )

    # cyclical encodings in [-1, 1]
    for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"):
        c.require(
            df[col].between(-1 - _EPS, 1 + _EPS).all(),
            f"{col} outside [-1, 1] or null",
        )

    # calendar columns: ranges, and agreement with pandas' own calendar
    ranges = {
        "month": (1, 12), "day": (1, 31), "hour": (0, 23), "quarter": (1, 4),
        "day_of_year": (1, 366), "week_of_year": (1, 53), "day_of_week": (0, 6),
    }
    for col, (lo, hi) in ranges.items():
        c.require(df[col].between(lo, hi).all(), f"{col} outside [{lo}, {hi}] or null")
    c.require(df["season"].isin(set(config.SEASON_BY_MONTH.values())).all(), "unknown season value")
    c.require((df["hour"] == df["ts_hour"].dt.hour).all(), "hour column does not match ts_hour")
    c.require((df["day_of_week"] == df["ts_hour"].dt.dayofweek).all(), "day_of_week (Mon=0) does not match ts_hour")
    c.require((df["is_weekend"] == (df["day_of_week"] >= 5)).all(), "is_weekend inconsistent with day_of_week")

    # lag columns: wherever the source hour exists in gold, the lag must equal its demand
    # (null if that demand is null). A lag may be null where the source hour is absent.
    demand_by_hour = full.set_index("ts_hour")["demand_mw"]
    for n in config.LAG_HOURS:
        source_ts = df["ts_hour"] - pd.Timedelta(hours=n)
        exists = source_ts.isin(demand_by_hour.index).to_numpy()
        expected = demand_by_hour.reindex(source_ts).to_numpy()
        actual = df[features.lag_col(n)].to_numpy()
        same = (np.isnan(expected) & np.isnan(actual)) | np.isclose(expected, actual, rtol=0, atol=_EPS, equal_nan=False)
        wrong = int((exists & ~same).sum())
        c.require(wrong == 0, f"{features.lag_col(n)}: {wrong} rows differ from demand at ts_hour - {n}h")

    # daily_agg: 24 hours per day, flags consistent, one row per day of hourly_master
    daily = pd.read_parquet(config.DAILY_AGG_DIR, filters=[("data_month", "in", list(months))])
    c.require(
        ((daily["hours_covered"] + daily["hours_missing"]) == 24).all(),
        "daily_agg: hours_covered + hours_missing != 24",
    )
    c.require(
        (daily["is_complete"] == (daily["hours_missing"] == 0)).all(),
        "daily_agg: is_complete inconsistent with hours_missing",
    )
    c.require(
        len(daily) == df["ts_hour"].dt.date.nunique(),
        f"daily_agg has {len(daily)} days but hourly_master has {df['ts_hour'].dt.date.nunique()}",
    )
    lf = daily["load_factor"].dropna()
    c.require(((lf > 0) & (lf <= 1 + _EPS)).all(), "daily_agg: load_factor outside (0, 1]")
    c.finish()