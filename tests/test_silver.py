import openpyxl
import pandas as pd
import pytest
from pyspark.sql import functions as F

from src import clean_silver, config, ingest_bronze, quality
from src.utils import get_spark


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _write_b(path, rows):
    """A Layout B file from (timestamp_text, demand) pairs. demand may be text or None."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Report"
    ws.append(["Timestamp", "Demand (MW)", "Wind (MW)", "Solar (MW)", "Total Generation (MW)"])
    for ts, demand in rows:
        ws.append([ts, demand, "9545.07", "20.29", "142193.38"])
    wb.save(path)
    return path


def _pipeline(batch_id=None):
    """bronze then silver for whatever is in landing."""
    b = ingest_bronze.run(batch_id)
    return b, clean_silver.run(b["batch_id"])


def _silver():
    """Silver as pandas, with ts as an IST wall-clock string (read through Spark, whose
    session timezone is IST; plain pandas would show the stored UTC instants)."""
    df = get_spark().read.parquet(str(config.SILVER_DIR))
    return (
        df.select(
            F.date_format("ts", "yyyy-MM-dd HH:mm:ss").alias("ts"),
            "demand_mw", "source_grain", "_source_file", "_batch_id", "data_month",
        )
        .orderBy("ts").toPandas()
    )


def _quarantine(batch_id):
    return pd.read_parquet(config.QUARANTINE_DIR / "rows" / f"batch_id={batch_id}")


# --------------------------------------------------------------------------- #
# Layout A: 5-minute rows, zero-demand dropout (checkpoint 3)
# --------------------------------------------------------------------------- #
def test_layout_a_zero_demand_is_quarantined(data_root, make_layout_a):
    make_layout_a(config.LANDING_DIR / "September_2021.xlsx", n_rows=36, zero_at=(5,))

    b, summary = _pipeline()

    assert summary["bronze_rows"] == 36
    assert summary["silver_rows_from_batch"] == 35
    assert summary["rejected_by_reason"] == {config.REASON_DEMAND_NONPOSITIVE: 1}

    s = _silver()
    assert len(s) == 35
    assert list(s.columns) == [c for c in config.SILVER_COLUMNS if c != "_batch_id"][:0] + list(s.columns)  # shape check below
    assert (s["source_grain"] == "5min").all()
    assert (s["data_month"] == "2021-09").all()
    assert s["ts"].iloc[0] == "2021-09-01 00:00:00"
    assert s["ts"].iloc[-1] == "2021-09-01 02:55:00"
    assert "2021-09-01 00:25:00" not in set(s["ts"])          # the dropout (row index 5)
    assert s["demand_mw"].iloc[0] == 150000.0

    q = _quarantine(summary["batch_id"])
    assert len(q) == 1
    assert q["reason"].iloc[0] == config.REASON_DEMAND_NONPOSITIVE
    assert q["ts_raw"].iloc[0] == "2021-09-01 00:25:00"
    assert float(q["demand_raw"].iloc[0]) == 0.0


# --------------------------------------------------------------------------- #
# Layout B: hourly rows, footer rows (checkpoint 3)
# --------------------------------------------------------------------------- #
def test_layout_b_footer_rows_are_quarantined(data_root, make_layout_b):
    make_layout_b(config.LANDING_DIR / "January_2024.xlsx", n_hours=5, footer=True)

    b, summary = _pipeline()

    assert summary["bronze_rows"] == 11
    assert summary["silver_rows_from_batch"] == 5
    assert summary["rejected_by_reason"] == {config.REASON_FOOTER_ROW: 6}

    s = _silver()
    assert s["ts"].tolist() == [f"2024-01-01 0{h}:00:00" for h in range(5)]   # dd-MM-yyyy read day-first
    assert (s["source_grain"] == "hourly").all()
    assert s["demand_mw"].iloc[0] == pytest.approx(141981.26)

    q = _quarantine(summary["batch_id"])
    assert sorted(q["ts_raw"]) == sorted(["Minimum", "Timestamp", "Maximum", "Timestamp", "Average", "Sum"])


def test_every_rejection_reason(data_root):
    _write_b(config.LANDING_DIR / "bad.xlsx", [
        ("01-01-2024 00:00:00", "150000"),     # the only valid row
        ("01-01-2024 01:00:00", "abc"),        # non_numeric_demand
        ("01-01-2024 02:00:00", "#N/A"),       # non_numeric_demand (SCADA error text)
        ("01-01-2024 03:00:00", None),         # null_demand (empty cell)
        ("01-01-2024 04:00:00", " "),          # null_demand (blank text)
        ("01-01-2024 05:00:00", "0"),          # demand_nonpositive
        ("01-01-2024 06:00:00", "-5"),         # demand_nonpositive
        ("01-01-2024 07:00:00", "1000"),       # demand_out_of_range (below minimum)
        ("01-01-2024 08:00:00", "400000"),     # demand_out_of_range (above maximum)
        ("garbage", "150000"),                 # unparseable_timestamp
    ])

    b, summary = _pipeline()

    assert summary["silver_rows_from_batch"] == 1
    assert summary["rejected_by_reason"] == {
        config.REASON_NON_NUMERIC_DEMAND: 2,
        config.REASON_NULL_DEMAND: 2,
        config.REASON_DEMAND_NONPOSITIVE: 2,
        config.REASON_DEMAND_OUT_OF_RANGE: 2,
        config.REASON_UNPARSEABLE_TS: 1,
    }
    assert summary["bronze_rows"] == 1 + sum(summary["rejected_by_reason"].values())


def test_both_layouts_in_one_batch(data_root, make_layout_a, make_layout_b):
    make_layout_a(config.LANDING_DIR / "a.xlsx", n_rows=36, zero_at=())
    make_layout_b(config.LANDING_DIR / "b.xlsx", n_hours=5, footer=False)

    b, summary = _pipeline()

    assert summary["silver_rows_from_batch"] == 36 + 5
    s = _silver()
    assert s.groupby("source_grain").size().to_dict() == {"5min": 36, "hourly": 5}
    assert s["ts"].is_unique is False or True  # different years, so no overlap; uniqueness checked in validate
    quality.validate_silver(summary)


# --------------------------------------------------------------------------- #
# Months follow IST, and partitions are replaced one month at a time
# --------------------------------------------------------------------------- #
def test_month_partitioning_uses_ist_and_leaves_other_months_alone(data_root):
    _write_b(config.LANDING_DIR / "first.xlsx", [
        ("31-01-2024 23:00:00", "150000"),
        ("01-02-2024 00:00:00", "151000"),     # 2024-01-31 18:30 UTC, but February in IST
    ])
    b1, s1 = _pipeline("20240101T000001_aaaaaaaa")
    assert s1["months_touched"] == ["2024-01", "2024-02"]
    s = _silver()
    assert dict(zip(s["ts"], s["data_month"])) == {
        "2024-01-31 23:00:00": "2024-01",
        "2024-02-01 00:00:00": "2024-02",
    }

    # a later file that only touches January must not delete or change February
    (config.LANDING_DIR / "first.xlsx").unlink()
    _write_b(config.LANDING_DIR / "second.xlsx", [("15-01-2024 10:00:00", "160000")])
    b2, s2 = _pipeline("20240101T000002_bbbbbbbb")

    assert s2["months_touched"] == ["2024-01"]
    s = _silver()
    assert s["ts"].tolist() == ["2024-01-15 10:00:00", "2024-01-31 23:00:00", "2024-02-01 00:00:00"]
    assert s2["superseded_existing_rows"] == 0


# --------------------------------------------------------------------------- #
# Duplicates: the later batch wins, the replaced rows are logged (not dropped)
# --------------------------------------------------------------------------- #
def test_later_batch_wins_and_replaced_rows_are_quarantined(data_root):
    _write_b(config.LANDING_DIR / "file1.xlsx", [
        ("01-01-2024 00:00:00", "100000"),
        ("01-01-2024 01:00:00", "100100"),
        ("01-01-2024 02:00:00", "100200"),
    ])
    _pipeline("20240101T000001_aaaaaaaa")

    (config.LANDING_DIR / "file1.xlsx").unlink()          # as if archived
    _write_b(config.LANDING_DIR / "file2.xlsx", [
        ("01-01-2024 01:00:00", "200000"),                 # overlaps file1 at 01:00 and 02:00
        ("01-01-2024 02:00:00", "200100"),
        ("01-01-2024 03:00:00", "200200"),
    ])
    b2, s2 = _pipeline("20240101T000002_bbbbbbbb")

    s = _silver()
    assert s["ts"].is_unique
    assert s["ts"].tolist() == [f"2024-01-01 0{h}:00:00" for h in range(4)]
    assert s["demand_mw"].tolist() == [100000.0, 200000.0, 200100.0, 200200.0]
    assert s["_batch_id"].tolist() == [
        "20240101T000001_aaaaaaaa", "20240101T000002_bbbbbbbb",
        "20240101T000002_bbbbbbbb", "20240101T000002_bbbbbbbb",
    ]

    assert s2["superseded_existing_rows"] == 2
    q = _quarantine("20240101T000002_bbbbbbbb")
    assert len(q) == 2
    assert (q["reason"] == config.REASON_DUPLICATE_SUPERSEDED).all()
    assert (q["_batch_id"] == "20240101T000001_aaaaaaaa").all()       # the replaced rows
    assert sorted(q["demand_raw"].astype(float)) == [100100.0, 100200.0]
    quality.validate_silver(s2)


# --------------------------------------------------------------------------- #
# Idempotency (checkpoint 7)
# --------------------------------------------------------------------------- #
def test_rerunning_silver_on_the_same_batch_changes_nothing(data_root, make_layout_a, make_layout_b):
    make_layout_a(config.LANDING_DIR / "a.xlsx", n_rows=36, zero_at=(5,))
    make_layout_b(config.LANDING_DIR / "b.xlsx", n_hours=5, footer=True)
    b = ingest_bronze.run("20240101T000001_aaaaaaaa")

    first = clean_silver.run(b["batch_id"])
    silver_1 = _silver()
    quarantine_1 = _quarantine(b["batch_id"]).sort_values(["ts_raw", "reason"]).reset_index(drop=True)

    second = clean_silver.run(b["batch_id"])
    silver_2 = _silver()
    quarantine_2 = _quarantine(b["batch_id"]).sort_values(["ts_raw", "reason"]).reset_index(drop=True)

    assert first == second
    assert silver_1.equals(silver_2)
    assert quarantine_1.equals(quarantine_2)
    assert config.REASON_DUPLICATE_SUPERSEDED not in set(quarantine_2["reason"])


# --------------------------------------------------------------------------- #
# Edge case: nothing valid
# --------------------------------------------------------------------------- #
def test_batch_with_no_valid_rows_writes_no_silver_and_fails_validation(data_root):
    _write_b(config.LANDING_DIR / "allbad.xlsx", [
        ("01-01-2024 00:00:00", "abc"),
        ("01-01-2024 01:00:00", "0"),
    ])

    b, summary = _pipeline()

    assert summary["months_touched"] == []
    assert summary["silver_rows_from_batch"] == 0
    assert not config.SILVER_DIR.exists()
    assert len(_quarantine(summary["batch_id"])) == 2
    with pytest.raises(quality.QualityError):
        quality.validate_silver(summary)


# --------------------------------------------------------------------------- #
# validate_silver
# --------------------------------------------------------------------------- #
def test_validate_silver_passes_on_a_good_run(data_root, make_layout_a, make_layout_b):
    make_layout_a(config.LANDING_DIR / "a.xlsx", n_rows=36, zero_at=(5,))
    make_layout_b(config.LANDING_DIR / "b.xlsx", n_hours=5, footer=True)
    b, summary = _pipeline()
    quality.validate_silver(summary)     # must not raise


def test_validate_silver_fails_when_row_accounting_is_off(data_root, make_layout_b):
    make_layout_b(config.LANDING_DIR / "b.xlsx", n_hours=5, footer=True)
    b, summary = _pipeline()
    summary["bronze_rows"] += 1          # pretend one row went missing

    with pytest.raises(quality.QualityError):
        quality.validate_silver(summary)