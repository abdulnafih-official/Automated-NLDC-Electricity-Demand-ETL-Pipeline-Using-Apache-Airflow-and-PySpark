"""Silver: bronze -> clean, typed, native grain (5-minute for Layout A, hourly for Layout B).

No resampling here. Every row that is not kept goes to quarantine/rows with a reason,
so: bronze rows of the batch = kept rows + quarantined rows.
"""

import shutil
from functools import reduce

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src import config
from src.utils import get_logger, get_spark

log = get_logger(__name__)



# --------------------------------------------------------------------------- #
# Per-layout cleaning
# --------------------------------------------------------------------------- #
def _classify(bronze_layout: DataFrame, layout_key: str) -> DataFrame:
    """Add parsed ts / demand_mw and a `reason` column (null = row is valid).
    The first matching rule wins, in the order of the plan."""
    layout = config.SOURCE_LAYOUTS[layout_key]

    ts_raw = F.col(layout["timestamp_col"])
    ts_str = F.trim(ts_raw)
    fmt = layout["timestamp_format_spark"]
    ts = F.to_timestamp(ts_str, fmt) if fmt else F.to_timestamp(ts_str)

    dem_raw = F.col(layout["demand_col"])
    dem_str = F.when(F.trim(dem_raw) == "", None).otherwise(F.trim(dem_raw))
    dem = dem_str.cast("double")

    df = bronze_layout.select(
        ts_raw.alias("ts_raw"),
        dem_raw.alias("demand_raw"),
        ts.alias("ts"),
        dem_str.alias("_dem_str"),
        dem.alias("demand_mw"),
        F.lit(layout["grain"]).alias("source_grain"),
        F.col("_source_file"),
        F.col("_source_layout"),
        F.col("_batch_id"),
        ts_str.alias("_ts_str"),
    )

    reason = None
    if layout["footer_markers"]:
        reason = F.when(F.col("_ts_str").isin(list(layout["footer_markers"])), config.REASON_FOOTER_ROW)
    rules = [
        (F.col("ts").isNull(), config.REASON_UNPARSEABLE_TS),
        (F.col("_dem_str").isNull(), config.REASON_NULL_DEMAND),
        (
            F.col("demand_mw").isNull()
            | F.isnan("demand_mw")
            | F.col("demand_mw").isin(float("inf"), float("-inf")),
            config.REASON_NON_NUMERIC_DEMAND,
        ),
        (F.col("demand_mw") <= 0, config.REASON_DEMAND_NONPOSITIVE),
        (
            (F.col("demand_mw") < config.DEMAND_VALID_MIN_MW)
            | (F.col("demand_mw") > config.DEMAND_VALID_MAX_MW),
            config.REASON_DEMAND_OUT_OF_RANGE,
        ),
    ]
    for condition, code in rules:
        reason = F.when(condition, code) if reason is None else reason.when(condition, code)

    return df.withColumn("reason", reason).drop("_dem_str", "_ts_str")


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run(batch_id: str = None) -> dict:
    """Clean the bronze rows of one batch and write silver.

    batch_id: the batch produced by ingest_bronze. If None, the most recent batch in bronze.
    """
    spark = get_spark()
    if not config.BRONZE_DIR.exists():
        raise FileNotFoundError(f"No bronze data at {config.BRONZE_DIR}")

    bronze = spark.read.option("mergeSchema", "true").parquet(str(config.BRONZE_DIR))
    if batch_id is None:
        batch_id = bronze.agg(F.max("_batch_id")).first()[0]
    run_batch_id = batch_id
    bronze = bronze.filter(F.col("_batch_id") == batch_id).cache()
    bronze_rows = bronze.count()
    if bronze_rows == 0:
        raise ValueError(f"No bronze rows found for batch {batch_id}")
    log.info("silver start | batch=%s | bronze rows=%d", batch_id, bronze_rows)

    # 1. classify every row, layout by layout
    present = [r[0] for r in bronze.select("_source_layout").distinct().collect()]
    classified = reduce(
        DataFrame.unionByName,
        [_classify(bronze.filter(F.col("_source_layout") == k), k) for k in sorted(present)],
    )
    valid = (
        classified.filter(F.col("reason").isNull())
        .withColumn("data_month", F.date_format("ts", "yyyy-MM"))
        .select(*config.SILVER_COLUMNS)
    )
    rejected = classified.filter(F.col("reason").isNotNull()).select(
        "ts_raw", "demand_raw", "reason", "_source_file", "_source_layout", "_batch_id"
    )

    # 2. merge with existing silver rows of the months this batch touches, so one month
    #    can come from several files and a rewrite of the month never loses earlier rows
    months = [r[0] for r in valid.select("data_month").distinct().collect()]
    merged = valid
    if months and config.SILVER_DIR.exists():
        existing = (
            spark.read.parquet(str(config.SILVER_DIR))
            .filter(F.col("data_month").isin(months))
            .filter(F.col("_batch_id") != run_batch_id)   # a rerun regenerates its own rows
            .select(*config.SILVER_COLUMNS)
        )
        merged = valid.unionByName(existing)

    # 3. one row per timestamp: the later batch wins (batch ids start with an IST timestamp,
    #    so they sort chronologically); every superseded row is quarantined, not dropped
    w = Window.partitionBy("ts").orderBy(
        F.col("_batch_id").desc(), F.col("_source_file").desc(), F.col("demand_mw").desc()
    )
    ranked = merged.withColumn("_rn", F.row_number().over(w))
    kept = ranked.filter("_rn = 1").drop("_rn")
    superseded = ranked.filter("_rn > 1").select(
        F.date_format("ts", "yyyy-MM-dd HH:mm:ss").alias("ts_raw"),
        F.col("demand_mw").cast("string").alias("demand_raw"),
        F.lit(config.REASON_DUPLICATE_SUPERSEDED).alias("reason"),
        "_source_file",
        F.lit(None).cast("string").alias("_source_layout"),
        "_batch_id",
    )

    # 4. materialise before writing: silver is both read (step 2) and overwritten (step 5)
    kept = kept.localCheckpoint(eager=True)
    rejected_all = rejected.unionByName(superseded).localCheckpoint(eager=True)

    kept_rows = kept.count()
    batch_kept = kept.filter(F.col("_batch_id") == run_batch_id).count()
    by_reason = {
        r["reason"]: r["n"]
        for r in rejected_all.filter(F.col("_batch_id") == run_batch_id)
        .groupBy("reason").agg(F.count("*").alias("n")).collect()
    }
    superseded_existing = rejected_all.filter(
        (F.col("reason") == config.REASON_DUPLICATE_SUPERSEDED) & (F.col("_batch_id") != run_batch_id)
    ).count()

    # 5. write silver (dynamic partition overwrite: only the touched months are replaced)
    if months:
        (
            kept.repartition("data_month")
            .write.mode("overwrite").partitionBy("data_month")
            .parquet(str(config.SILVER_DIR))
        )

    # 6. write this batch's quarantined rows (a retry of the same batch replaces them)
    q_dir = config.QUARANTINE_DIR / "rows" / f"batch_id={run_batch_id}"
    shutil.rmtree(q_dir, ignore_errors=True)
    if rejected_all.count() > 0:
        rejected_all.coalesce(1).write.mode("overwrite").parquet(str(q_dir))

    summary = {
        "batch_id": run_batch_id,
        "bronze_rows": bronze_rows,
        "silver_rows_from_batch": batch_kept,
        "silver_rows_total_in_touched_months": kept_rows,
        "rejected_by_reason": by_reason,
        "superseded_existing_rows": superseded_existing,
        "months_touched": sorted(months),
    }
    log.info(
        "silver done | bronze=%d kept_from_batch=%d rejected=%d %s | months=%s",
        bronze_rows, batch_kept, sum(by_reason.values()), by_reason, sorted(months),
    )
    return summary