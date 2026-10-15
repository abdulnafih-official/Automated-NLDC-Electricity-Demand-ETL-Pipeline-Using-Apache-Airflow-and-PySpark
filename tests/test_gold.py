"""Gold layer tests: hourly resample, spine, features, daily_agg, idempotency, quality gate.

End-to-end tests go landing -> bronze -> silver -> gold using the small Excel fixtures from
conftest.py. Feature tests build a synthetic hourly frame in Spark (so timestamps are
parsed in the IST session timezone, like build_spine does).
"""

import math

import pandas as pd
import pytest
from pyspark.sql import functions as F

from src import build_gold, clean_silver, config, features, ingest_bronze, quality
from src.utils import get_spark


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _gold(months=None):
    """bronze -> silver -> gold for whatever is in landing. Returns the gold summary."""
    b = ingest_bronze.run()
    s = clean_silver.run(b["batch_id"])
    return build_gold.run(months or s["months_touched"])


def _hourly():
    df = pd.read_parquet(config.HOURLY_MASTER_DIR)
    df["data_month"] = df["data_month"].astype(str)
    return df.sort_values("ts_hour").reset_index(drop=True)


def _daily():
    df = pd.read_parquet(config.DAILY_AGG_DIR)
    df["data_month"] = df["data_month"].astype(str)
    return df.sort_values("date").reset_index(drop=True)


def _at(df, ts):
    return df[df["ts_hour"] == pd.Timestamp(ts)].iloc[0]


def _synthetic(n_hours, bump_from=None):
    """n_hours of hourly rows from 2024-01-01 00:00 IST. If bump_from is set, demand from
    that row index on is changed (used by the leakage test)."""
    spark = get_spark()
    demand = 100000.0 + F.col("id") * 10 + (F.col("id") % 24) * 500
    if bump_from is not None:
        demand = F.when(F.col("id") >= bump_from, demand + 99999.0).otherwise(demand)
    return (
        spark.range(n_hours)
        .withColumn(
            "ts_hour",
            (F.lit("2024-01-01 00:00:00").cast("timestamp").cast("long") + F.col("id") * 3600).cast("timestamp"),
        )
        .withColumn("demand_mw", demand.cast("double"))
        .withColumn("readings_count", F.lit(1))
        .withColumn("source_grain", F.lit("hourly"))
        .withColumn("data_month", F.date_format("ts_hour", "yyyy-MM"))
        .select(*features.CORE_COLUMNS, "data_month")
    )


# --------------------------------------------------------------------------- #
# Month arithmetic (pure)
# --------------------------------------------------------------------------- #
def test_months_to_rebuild_adds_following_month_only_if_it_exists():
    assert build_gold.months_to_rebuild(["2024-01"], {"2024-01", "2024-02"}) == ["2024-01", "2024-02"]
    assert build_gold.months_to_rebuild(["2024-01"], {"2024-01", "2024-03"}) == ["2024-01"]
    assert build_gold.months_to_rebuild(["2023-12"], {"2023-12", "2024-01"}) == ["2023-12", "2024-01"]


# --------------------------------------------------------------------------- #
# Hourly resample + spine (checkpoint 4)
# --------------------------------------------------------------------------- #
def test_layout_a_hour_is_mean_of_valid_readings(data_root, make_layout_a):
    # rows 0..35 = 3 hours; row 5 is a zero-demand dropout and is quarantined in silver
    make_layout_a(config.LANDING_DIR / "a.xlsx", n_rows=36, zero_at=(5,))

    summary = _gold()
    h = _hourly()

    hour0 = [150000.0 + i * 10 for i in range(12) if i != 5]      # 11 valid readings
    hour1 = [150000.0 + i * 10 for i in range(12, 24)]            # 12 readings
    r0, r1 = _at(h, "2021-09-01 00:00"), _at(h, "2021-09-01 01:00")
    assert r0["demand_mw"] == pytest.approx(sum(hour0) / 11)
    assert r0["readings_count"] == 11
    assert r1["demand_mw"] == pytest.approx(sum(hour1) / 12)
    assert r1["readings_count"] == 12
    assert (h["source_grain"] == "5min").all()

    # gap report: 1 day, 24 rows, hours 3..23 have no data -> 21 null hours
    g = summary["gap_report"][0]
    assert (g["data_month"], g["days"], g["hours"], g["null_demand_hours"]) == ("2021-09", 1, 24, 21)


def test_hour_with_too_few_readings_is_null(data_root, make_layout_a):
    # zero out rows 0..3 -> hour 0 has 8 valid readings, below MIN_READINGS_PER_HOUR (9)
    make_layout_a(config.LANDING_DIR / "a.xlsx", n_rows=24, zero_at=(0, 1, 2, 3))

    _gold()
    r = _at(_hourly(), "2021-09-01 00:00")

    assert r["readings_count"] == 8
    assert pd.isna(r["demand_mw"])


def test_spine_has_24_rows_per_covered_day_and_no_invented_days(data_root, make_layout_a, make_layout_b):
    make_layout_a(config.LANDING_DIR / "a.xlsx", n_rows=36, zero_at=())
    make_layout_b(config.LANDING_DIR / "b.xlsx", n_hours=5, footer=False)

    _gold()
    h = _hourly()

    assert len(h) == 48                                           # 2 covered days x 24
    assert h["ts_hour"].is_unique
    assert set(h["ts_hour"].dt.date.astype(str)) == {"2021-09-01", "2024-01-01"}
    assert h.groupby(h["ts_hour"].dt.date).size().eq(24).all()

    # missing hours stay as rows, with null demand and 0 readings, never imputed
    empty = _at(h, "2024-01-01 10:00")
    assert pd.isna(empty["demand_mw"]) and empty["readings_count"] == 0
    assert empty["source_grain"] == "hourly"                      # filled from the day's grain


# --------------------------------------------------------------------------- #
# Features (checkpoint 5)
# --------------------------------------------------------------------------- #
def test_calendar_features(data_root):
    df = features.add_all_features(_synthetic(24 * 7)).toPandas()
    df = df.sort_values("ts_hour").reset_index(drop=True)

    mon = df.iloc[0]                       # 2024-01-01 00:00, a Monday
    assert (mon["year"], mon["month"], mon["day"], mon["hour"]) == (2024, 1, 1, 0)
    assert (mon["day_of_week"], mon["is_weekend"], mon["season"], mon["quarter"]) == (0, False, "winter", 1)
    assert mon["hour_sin"] == pytest.approx(0.0) and mon["hour_cos"] == pytest.approx(1.0)

    sat = df.iloc[5 * 24 + 6]              # 2024-01-06 06:00, a Saturday
    assert (sat["day_of_week"], sat["is_weekend"], sat["hour"]) == (5, True, 6)
    assert sat["hour_sin"] == pytest.approx(1.0)
    assert sat["dow_sin"] == pytest.approx(math.sin(5 * 2 * math.pi / 7))


def test_lag_and_rolling_are_timestamp_based(data_root, make_layout_b):
    # 48 consecutive hours: row i has demand 141981.26 + 100 * i
    make_layout_b(config.LANDING_DIR / "b.xlsx", n_hours=48, footer=False)

    _gold()
    h = _hourly()
    d = lambda i: 141981.26 + 100 * i

    assert pd.isna(_at(h, "2024-01-01 00:00")["demand_lag_1h"])
    assert _at(h, "2024-01-01 01:00")["demand_lag_1h"] == pytest.approx(d(0))
    assert _at(h, "2024-01-01 03:00")["demand_lag_3h"] == pytest.approx(d(0))

    # lag_24h: null in the first day, equals the value 24 h earlier afterwards
    assert pd.isna(_at(h, "2024-01-01 23:00")["demand_lag_24h"])
    assert _at(h, "2024-01-02 05:00")["demand_lag_24h"] == pytest.approx(d(5))
    assert pd.isna(_at(h, "2024-01-02 05:00")["demand_lag_48h"])  # 48 h back does not exist

    # rolling 24h needs the FULL window of 24 prior hours, so first value is at row 24
    assert pd.isna(_at(h, "2024-01-01 23:00")["demand_roll_mean_24h"])
    r = _at(h, "2024-01-02 00:00")                                # window = rows 0..23
    assert r["demand_roll_mean_24h"] == pytest.approx(sum(d(i) for i in range(24)) / 24)
    assert r["demand_roll_min_24h"] == pytest.approx(d(0))
    assert r["demand_roll_max_24h"] == pytest.approx(d(23))
    assert pd.isna(_at(h, "2024-01-02 23:00")["demand_roll_mean_168h"])   # no 7 days of history

    # change features come from lags only
    r = _at(h, "2024-01-01 05:00")
    assert r["demand_ramp_1h"] == pytest.approx(d(4) - d(3))
    assert _at(h, "2024-01-02 05:00")["demand_diff_24h"] != _at(h, "2024-01-02 05:00")["demand_diff_24h"]  # NaN: lag_48h absent


def test_features_are_null_across_gap_between_files(data_root, make_layout_a, make_layout_b):
    make_layout_a(config.LANDING_DIR / "a.xlsx", n_rows=288, zero_at=())      # all of 2021-09-01
    make_layout_b(config.LANDING_DIR / "b.xlsx", n_hours=24, footer=False)    # all of 2024-01-01

    _gold()
    h = _hourly()
    first = h[h["ts_hour"].dt.date.astype(str) == "2024-01-01"]

    # 2.3 years of missing data must not leak into 2024: every lag and rolling value is null
    lag_cols = [features.lag_col(n) for n in config.LAG_HOURS]
    roll_cols = [c for c in h.columns if c.startswith("demand_roll_")]
    assert first[lag_cols[1:]].isna().all().all()                  # 2h..168h: nothing 2+ h back
    assert first[roll_cols].isna().all().all()
    assert first[features.same_hour_col()].isna().all()


def test_no_leakage_features_ignore_current_and_future_demand(data_root):
    """Changing demand at hour T and after must not change any feature at T."""
    T = 24 * 9 + 6                                                  # row 222 = 2024-01-10 06:00
    base = features.add_all_features(_synthetic(24 * 12))
    bumped = features.add_all_features(_synthetic(24 * 12, bump_from=T))

    row_a = base.orderBy("ts_hour").collect()[T].asDict()
    row_b = bumped.orderBy("ts_hour").collect()[T].asDict()
    row_a.pop("demand_mw"), row_b.pop("demand_mw")

    assert row_a == row_b
    assert row_a["demand_lag_24h"] is not None                      # sanity: features exist at T
    assert row_a["demand_roll_mean_168h"] is not None
    assert row_a[features.same_hour_col()] is not None


# --------------------------------------------------------------------------- #
# daily_agg
# --------------------------------------------------------------------------- #
def test_daily_agg_for_a_complete_day(data_root, make_layout_b):
    make_layout_b(config.LANDING_DIR / "b.xlsx", n_hours=24, footer=False)   # demand rises each hour

    _gold()
    d = _daily()
    vals = [141981.26 + 100 * i for i in range(24)]

    assert len(d) == 1
    r = d.iloc[0]
    assert (r["hours_covered"], r["hours_missing"], bool(r["is_complete"])) == (24, 0, True)
    assert r["demand_mean_mw"] == pytest.approx(sum(vals) / 24)
    assert (r["demand_peak_mw"], r["peak_hour"]) == (pytest.approx(vals[-1]), 23)
    assert (r["demand_min_mw"], r["min_hour"]) == (pytest.approx(vals[0]), 0)
    assert r["energy_mwh"] == pytest.approx(sum(vals))
    assert r["load_factor"] == pytest.approx((sum(vals) / 24) / vals[-1])


def test_daily_agg_for_an_incomplete_day(data_root, make_layout_b):
    make_layout_b(config.LANDING_DIR / "b.xlsx", n_hours=5, footer=False)

    _gold()
    r = _daily().iloc[0]

    assert (r["hours_covered"], r["hours_missing"], bool(r["is_complete"])) == (5, 19, False)


# --------------------------------------------------------------------------- #
# Idempotency (checkpoint 7)
# --------------------------------------------------------------------------- #
def test_rerunning_gold_gives_identical_output(data_root, make_layout_b):
    make_layout_b(config.LANDING_DIR / "b.xlsx", n_hours=48, footer=False)
    b = ingest_bronze.run()
    s = clean_silver.run(b["batch_id"])

    build_gold.run(s["months_touched"])
    first_h, first_d = _hourly(), _daily()
    build_gold.run(s["months_touched"])
    second_h, second_d = _hourly(), _daily()

    pd.testing.assert_frame_equal(first_h, second_h)
    pd.testing.assert_frame_equal(first_d, second_d)
    assert second_h["ts_hour"].is_unique


def test_unknown_month_is_rejected(data_root, make_layout_b):
    make_layout_b(config.LANDING_DIR / "b.xlsx", n_hours=5, footer=False)
    b = ingest_bronze.run()
    clean_silver.run(b["batch_id"])

    with pytest.raises(ValueError, match="not found in silver"):
        build_gold.run(["1999-01"])


# --------------------------------------------------------------------------- #
# Quality gate
# --------------------------------------------------------------------------- #
def test_validate_gold_passes_on_clean_data(data_root, make_layout_b):
    make_layout_b(config.LANDING_DIR / "b.xlsx", n_hours=48, footer=False)

    summary = _gold()

    quality.validate_gold(summary)                                  # must not raise


def test_validate_gold_fails_when_too_many_hours_are_null(data_root, make_layout_a):
    # 3 hours of data in a 24-hour day -> 21 of 24 hours null, far above MAX_NULL_DEMAND_RATE
    make_layout_a(config.LANDING_DIR / "a.xlsx", n_rows=36, zero_at=())

    summary = _gold()

    with pytest.raises(quality.QualityError, match="null rate"):
        quality.validate_gold(summary)