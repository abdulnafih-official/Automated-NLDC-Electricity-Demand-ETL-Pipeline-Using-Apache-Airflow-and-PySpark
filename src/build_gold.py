"""Gold: silver -> hourly resample -> hourly spine -> features -> hourly_master + daily_agg.

Plan section 6:
  1. resample: demand_mw = mean of valid readings in the hour; null if too few readings
  2. spine: one row for EVERY hour of every day that has any silver data (no invented days)
  3. features: calendar, lag, rolling, change (src/features.py), leakage-safe
Plan section 8: reads LOOKBACK_DAYS before each rebuilt month, and also rebuilds the
following month when it exists, so features stay correct on reruns and backfills.
"""

from datetime import date, timedelta
from functools import reduce
from operator import or_

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src import config, features
from src.utils import get_logger, get_spark

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Month arithmetic
# --------------------------------------------------------------------------- #
def _month_start(month: str) -> date:
    year, mon = (int(x) for x in month.split("-"))
    return date(year, mon, 1)


def _next_month(month: str) -> str:
    start = _month_start(month)
    nxt = date(start.year + (start.month == 12), start.month % 12 + 1, 1)
    return f"{nxt:%Y-%m}"


def months_to_rebuild(target: list, available: set) -> list:
    """Target months plus the following month of each, if that month exists in silver."""
    rebuild = set(target)
    for m in target:
        if _next_month(m) in available:
            rebuild.add(_next_month(m))
    return sorted(rebuild)


# --------------------------------------------------------------------------- #
# Step 1: resample to hourly
# --------------------------------------------------------------------------- #
def resample_hourly(silver: DataFrame) -> DataFrame:
    """One row per hour that has at least one silver reading.

    5-minute sources need MIN_READINGS_PER_HOUR valid readings, otherwise demand_mw is
    null (readings_count still shows how many were there). Hourly sources pass through.
    """
    needed = F.when(F.col("source_grain") == "5min", F.lit(config.MIN_READINGS_PER_HOUR)).otherwise(F.lit(1))
    return (
        silver.withColumn("ts_hour", F.date_trunc("hour", F.col("ts")))  # truncates in IST (session tz)
        .groupBy("ts_hour")
        .agg(
            F.avg("demand_mw").alias("_mean"),
            F.count("*").cast("int").alias("readings_count"),
            F.max("source_grain").alias("source_grain"),
        )
        .withColumn("demand_mw", F.when(F.col("readings_count") >= needed, F.col("_mean")))
        .drop("_mean")
    )


# --------------------------------------------------------------------------- #
# Step 2: hourly spine
# --------------------------------------------------------------------------- #
def build_spine(hourly: DataFrame) -> DataFrame:
    """24 rows for every day that has any data. Hours without usable demand stay as rows
    with null demand_mw and readings_count 0. Days with no data at all get no rows."""
    days = hourly.select(F.to_date("ts_hour").alias("d")).distinct()
    spine = (
        days.select("d", F.explode(F.sequence(F.lit(0), F.lit(23))).alias("h"))
        .withColumn(
            "ts_hour",
            (F.col("d").cast("timestamp").cast("long") + F.col("h") * 3600).cast("timestamp"),
        )
        .select("d", "ts_hour")
    )
    day_grain = hourly.groupBy(F.to_date("ts_hour").alias("d")).agg(F.max("source_grain").alias("_day_grain"))

    return (
        spine.join(hourly, "ts_hour", "left")
        .join(day_grain, "d", "left")
        .withColumn("readings_count", F.coalesce(F.col("readings_count"), F.lit(0)))
        .withColumn("source_grain", F.coalesce(F.col("source_grain"), F.col("_day_grain")))
        .withColumn("data_month", F.date_format("ts_hour", "yyyy-MM"))
        .select(*features.CORE_COLUMNS, "data_month")
    )


# --------------------------------------------------------------------------- #
# daily_agg
# --------------------------------------------------------------------------- #
def build_daily_agg(base: DataFrame) -> DataFrame:
    """One row per day from the hourly rows (before features are added).
    peak_hour / min_hour are the hour of day (0-23)."""
    return (
        base.withColumn("date", F.to_date("ts_hour"))
        .withColumn("_hour", F.hour("ts_hour"))
        .groupBy("date", "data_month")
        .agg(
            F.count("demand_mw").cast("int").alias("hours_covered"),
            F.avg("demand_mw").alias("demand_mean_mw"),
            F.max("demand_mw").alias("demand_peak_mw"),
            F.max_by("_hour", "demand_mw").alias("peak_hour"),
            F.min("demand_mw").alias("demand_min_mw"),
            F.min_by("_hour", "demand_mw").alias("min_hour"),
            F.sum("demand_mw").alias("energy_mwh"),  # hourly mean MW x 1 h = MWh
        )
        .withColumn("hours_missing", (F.lit(24) - F.col("hours_covered")).cast("int"))
        .withColumn("load_factor", F.col("demand_mean_mw") / F.col("demand_peak_mw"))
        .withColumn("is_complete", F.col("hours_missing") == 0)
        .select(
            "date", "hours_covered", "hours_missing", "demand_mean_mw",
            "demand_peak_mw", "peak_hour", "demand_min_mw", "min_hour",
            "load_factor", "energy_mwh", "is_complete", "data_month",
        )
    )


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def _write_partitioned(df: DataFrame, path) -> None:
    """Dynamic partition overwrite (set in get_spark): only the months in df are replaced."""
    (
        df.repartition("data_month")
        .write.mode("overwrite").partitionBy("data_month")
        .parquet(str(path))
    )


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run(months: list = None) -> dict:
    """Build gold.

    months: data_month values (YYYY-MM) whose silver changed, e.g. `months_touched` from
            clean_silver.run(). If None, every month in silver is rebuilt.
    """
    spark = get_spark()
    if not config.SILVER_DIR.exists():
        raise FileNotFoundError(f"No silver data at {config.SILVER_DIR}")

    silver_all = spark.read.parquet(str(config.SILVER_DIR))
    available = sorted(r[0] for r in silver_all.select("data_month").distinct().collect())
    target = sorted(months) if months else available
    unknown = set(target) - set(available)
    if unknown:
        raise ValueError(f"Months not found in silver: {sorted(unknown)}")
    rebuild = months_to_rebuild(target, set(available))
    log.info("gold start | target=%s | rebuilding=%s", target, rebuild)

    # Read the months to rebuild plus LOOKBACK_DAYS before each, so the first days of a
    # month still see the previous week's demand for lag / rolling features.
    ranges = []
    for m in rebuild:
        start = _month_start(m) - timedelta(days=config.LOOKBACK_DAYS)
        end = _month_start(_next_month(m))
        ranges.append(
            (F.col("ts") >= F.lit(f"{start} 00:00:00").cast("timestamp"))
            & (F.col("ts") < F.lit(f"{end} 00:00:00").cast("timestamp"))
        )
    silver = silver_all.filter(reduce(or_, ranges))

    hourly = resample_hourly(silver)
    base = build_spine(hourly).cache()
    featured = features.add_all_features(base)

    hourly_master = featured.filter(F.col("data_month").isin(rebuild)).cache()
    daily_agg = build_daily_agg(base.filter(F.col("data_month").isin(rebuild)))

    # gap report: what was built per month, and how many hours have no usable demand
    gap_report = [
        {
            "data_month": r["data_month"],
            "hours": r["hours"],
            "days": r["hours"] // 24,
            "null_demand_hours": r["null_hours"],
        }
        for r in hourly_master.groupBy("data_month")
        .agg(F.count("*").alias("hours"), F.sum(F.col("demand_mw").isNull().cast("int")).alias("null_hours"))
        .orderBy("data_month").collect()
    ]
    for g in gap_report:
        log.info(
            "gold month %s | days=%d hours=%d null_demand_hours=%d",
            g["data_month"], g["days"], g["hours"], g["null_demand_hours"],
        )

    # ts_hour is written as local wall-clock time (no UTC adjustment), so pandas and
    # Power BI show 2024-01-01 00:00 as 00:00 instead of shifting it to UTC.
    hourly_out = hourly_master.withColumn("ts_hour", F.col("ts_hour").cast("timestamp_ntz"))
    _write_partitioned(hourly_out, config.HOURLY_MASTER_DIR)
    _write_partitioned(daily_agg, config.DAILY_AGG_DIR)

    total_rows = sum(g["hours"] for g in gap_report)
    log.info("gold done | months=%s | hourly_master rows=%d", rebuild, total_rows)
    return {
        "months_rebuilt": rebuild,
        "hourly_master_rows": total_rows,
        "gap_report": gap_report,
    }