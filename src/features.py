"""Pure feature functions for hourly_master (plan section 6, gold step 3).

Input: a Spark DataFrame with ONE ROW PER HOUR of every covered day, with columns
    ts_hour (timestamp, unique), demand_mw (double, nullable), readings_count,
    source_grain, data_month
Output: the same rows with calendar, lag, rolling and change features added, in the
column order of the hourly_master schema.

Rules (plan section 6):
  - Lags are timestamp-based: demand at ts_hour minus N hours, found by time, never by
    row position. If that hour is absent or null, the feature is null.
  - Rolling windows end at ts_hour - 1h and need the FULL window non-null, else null.
  - No feature uses demand_mw of its own row, so demand_mw can be the prediction target.
  - Nothing is imputed.
"""

import math
from functools import reduce
from itertools import chain
from operator import add

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src import config

_EPOCH = "_ts_epoch"  # ts_hour in seconds; range windows need a numeric ordering column

CORE_COLUMNS = ["ts_hour", "demand_mw", "readings_count", "source_grain"]
CALENDAR_COLUMNS = [
    "year", "month", "day", "hour", "quarter", "day_of_year", "week_of_year",
    "day_of_week", "is_weekend", "season",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
]


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #
def lag_col(n_hours: int) -> str:
    return f"demand_lag_{n_hours}h"


def roll_col(stat: str, window_hours: int) -> str:
    return f"demand_roll_{stat}_{window_hours}h"


def same_hour_col() -> str:
    return f"demand_same_hour_mean_{config.SAME_HOUR_LOOKBACK_DAYS}d"


def feature_columns() -> list:
    """Final hourly_master column order (plan section 6 schema). data_month is last."""
    rolling = [
        roll_col(stat, w)
        for w, stats in config.ROLLING_WINDOWS_HOURS.items()
        for stat in stats
    ]
    return (
        CORE_COLUMNS
        + CALENDAR_COLUMNS
        + [lag_col(n) for n in config.LAG_HOURS]
        + rolling
        + ["demand_ramp_1h", "demand_diff_24h", same_hour_col()]
        + ["data_month"]
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _offset_window(start_hours: int, end_hours: int) -> Window:
    """Rows whose ts_hour lies between ts_hour+start and ts_hour+end (inclusive).
    Negative = the past. Chosen by TIME, so missing hours cannot shift the match."""
    return Window.orderBy(_EPOCH).rangeBetween(start_hours * 3600, end_hours * 3600)


def _same_hour_lags() -> list:
    return [24 * d for d in range(1, config.SAME_HOUR_LOOKBACK_DAYS + 1)]


# --------------------------------------------------------------------------- #
# Feature groups
# --------------------------------------------------------------------------- #
def add_calendar_features(df: DataFrame) -> DataFrame:
    ts = F.col("ts_hour")
    season_map = F.create_map(
        *chain.from_iterable((F.lit(m), F.lit(s)) for m, s in config.SEASON_BY_MONTH.items())
    )
    return (
        df.withColumn("year", F.year(ts))
        .withColumn("month", F.month(ts))
        .withColumn("day", F.dayofmonth(ts))
        .withColumn("hour", F.hour(ts))
        .withColumn("quarter", F.quarter(ts))
        .withColumn("day_of_year", F.dayofyear(ts))
        .withColumn("week_of_year", F.weekofyear(ts))                     # ISO week
        .withColumn("day_of_week", (F.dayofweek(ts) + 5) % 7)             # Spark: Sun=1 -> Mon=0 ... Sun=6
        .withColumn("is_weekend", F.col("day_of_week") >= 5)
        .withColumn("season", season_map[F.col("month")])
        .withColumn("hour_sin", F.sin(F.col("hour") * (2 * math.pi / 24)))
        .withColumn("hour_cos", F.cos(F.col("hour") * (2 * math.pi / 24)))
        .withColumn("dow_sin", F.sin(F.col("day_of_week") * (2 * math.pi / 7)))
        .withColumn("dow_cos", F.cos(F.col("day_of_week") * (2 * math.pi / 7)))
        .withColumn("month_sin", F.sin((F.col("month") - 1) * (2 * math.pi / 12)))
        .withColumn("month_cos", F.cos((F.col("month") - 1) * (2 * math.pi / 12)))
    )


def add_lag_features(df: DataFrame) -> DataFrame:
    """demand at ts_hour - N hours. Also adds the daily lags (24h ... 168h) that the
    same-hour mean needs; add_all_features drops the ones not in LAG_HOURS."""
    needed = sorted(set(config.LAG_HOURS) | set(_same_hour_lags()))
    for n in needed:
        # The window holds only the row exactly N hours back (or nothing -> null).
        df = df.withColumn(lag_col(n), F.max("demand_mw").over(_offset_window(-n, -n)))
    return df


def add_rolling_features(df: DataFrame) -> DataFrame:
    """Statistics over the W hours ending at ts_hour - 1h. Null unless all W hours exist
    and are non-null. Std is the sample standard deviation (n - 1)."""
    stat_fn = {"mean": F.avg, "std": F.stddev, "min": F.min, "max": F.max}
    for w, stats in config.ROLLING_WINDOWS_HOURS.items():
        win = _offset_window(-w, -1)
        cnt = f"_cnt_{w}h"
        df = df.withColumn(cnt, F.count("demand_mw").over(win))  # counts non-null values only
        for stat in stats:
            df = df.withColumn(
                roll_col(stat, w),
                F.when(F.col(cnt) == w, stat_fn[stat]("demand_mw").over(win)),
            )
        df = df.drop(cnt)
    return df


def add_change_features(df: DataFrame) -> DataFrame:
    """Built from lagged values only, so they never touch the current hour's demand."""
    return df.withColumn(
        "demand_ramp_1h", F.col(lag_col(1)) - F.col(lag_col(2))
    ).withColumn(
        "demand_diff_24h", F.col(lag_col(24)) - F.col(lag_col(48))
    )


def add_same_hour_mean(df: DataFrame) -> DataFrame:
    """Mean of demand at the same clock hour on each of the previous N days.
    Null unless all N values exist."""
    cols = [F.col(lag_col(n)) for n in _same_hour_lags()]
    all_present = reduce(lambda a, b: a & b, [c.isNotNull() for c in cols])
    return df.withColumn(same_hour_col(), F.when(all_present, reduce(add, cols) / len(cols)))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def add_all_features(df: DataFrame) -> DataFrame:
    required = {1, 2, 24, 48}  # used by the ramp and diff features
    missing = required - set(config.LAG_HOURS)
    if missing:
        raise ValueError(f"config.LAG_HOURS must include {sorted(required)}; missing {sorted(missing)}")

    df = df.withColumn(_EPOCH, F.col("ts_hour").cast("long"))
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_change_features(df)
    df = add_same_hour_mean(df)
    return df.select(*feature_columns())  # also drops _ts_epoch and the extra daily lags