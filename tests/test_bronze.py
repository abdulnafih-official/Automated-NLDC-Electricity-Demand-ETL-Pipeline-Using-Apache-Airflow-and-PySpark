import hashlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src import config, ingest_bronze, quality


def _bronze_files():
    return sorted(config.BRONZE_DIR.glob("ingest_date=*/*.parquet"))


def _md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Small pure functions
# --------------------------------------------------------------------------- #
def test_sanitize_column_name():
    assert config.sanitize_column_name("Demand (MW)") == "Demand_MW"
    assert config.sanitize_column_name("NLDC_DEMAND|P") == "NLDC_DEMAND_P"
    assert config.sanitize_column_name("Solar+Wind") == "Solar_Wind"
    assert config.sanitize_column_name("SPARE 05") == "SPARE_05"


def test_build_column_names_prefers_lower_row_then_falls_back():
    header = pd.DataFrame([
        ["Time", "TAG1", None, None, "X", "X"],
        [None, "Name 1", None, None, "X", "X"],
    ])
    assert ingest_bronze.build_column_names(header) == [
        "Time", "Name_1", "col_3", "col_4", "X", "X_6",
    ]


def test_detect_layout(data_root, make_layout_a, make_layout_b, make_unknown):
    a = make_layout_a(config.LANDING_DIR / "a.xlsx")
    b = make_layout_b(config.LANDING_DIR / "b.xlsx")
    u = make_unknown(config.LANDING_DIR / "u.xlsx")
    assert ingest_bronze.detect_layout(a) == "A"
    assert ingest_bronze.detect_layout(b) == "B"
    assert ingest_bronze.detect_layout(u) is None


def test_count_excel_data_rows(data_root, make_layout_a, make_layout_b):
    a = make_layout_a(config.LANDING_DIR / "a.xlsx", n_rows=36)
    b = make_layout_b(config.LANDING_DIR / "b.xlsx", n_hours=5)
    assert ingest_bronze.count_excel_data_rows(a, "A") == 36
    assert ingest_bronze.count_excel_data_rows(b, "B") == 5 + 6  # footer rows included


# --------------------------------------------------------------------------- #
# Checkpoint 2: bronze is 1:1 with the Excel file
# --------------------------------------------------------------------------- #
def test_layout_a_bronze_is_one_to_one(data_root, make_layout_a):
    make_layout_a(config.LANDING_DIR / "September_2021.xlsx", n_rows=36, zero_at=(5,))

    summary = ingest_bronze.run()

    [item] = summary["ingested"]
    assert item["layout"] == "A"
    assert item["rows_written"] == item["expected_rows"] == 36

    df = pd.read_parquet(item["parquet"])
    assert len(df) == 36
    for col in ("Time", "NLDC_DEMAND_P", "Solar_Wind", "SPARE_05"):
        assert col in df.columns
    assert set(config.BRONZE_META_COLUMNS) <= set(df.columns)
    assert (df["_source_layout"] == "A").all()
    assert (df["_source_file"] == "September_2021.xlsx").all()
    assert df["_batch_id"].nunique() == 1
    assert df["Time"].iloc[0] == "2021-09-01 00:00:00"
    # bronze does not clean: the zero-demand dropout is still there
    assert float(df["NLDC_DEMAND_P"].iloc[5]) == 0.0


def test_layout_b_bronze_keeps_footer_rows_and_text_values(data_root, make_layout_b):
    make_layout_b(config.LANDING_DIR / "January_2024.xlsx", n_hours=5)

    [item] = ingest_bronze.run()["ingested"]

    assert item["layout"] == "B"
    assert item["rows_written"] == item["expected_rows"] == 5 + 6
    df = pd.read_parquet(item["parquet"])
    assert df["Timestamp"].tolist()[:1] == ["01-01-2024 00:00:00"]       # text, not parsed
    assert df["Demand_MW"].iloc[0] == "141981.26"
    assert df["Timestamp"].tail(6).tolist() == [
        "Minimum", "Timestamp", "Maximum", "Timestamp", "Average", "Sum",
    ]


def test_all_source_columns_are_strings(data_root, make_layout_a, make_layout_b):
    make_layout_a(config.LANDING_DIR / "a.xlsx")
    make_layout_b(config.LANDING_DIR / "b.xlsx")

    summary = ingest_bronze.run()

    for item in summary["ingested"]:
        schema = pq.read_schema(item["parquet"])
        source_cols = [c for c in schema.names if c not in config.BRONZE_META_COLUMNS]
        for col in source_cols:
            t = schema.field(col).type
            assert pa.types.is_string(t) or pa.types.is_large_string(t), f"{col} is {t}"


def test_landing_files_are_never_modified(data_root, make_layout_a):
    path = make_layout_a(config.LANDING_DIR / "a.xlsx")
    before = _md5(path)

    ingest_bronze.run()

    assert path.exists()              # bronze does not move it; archive_landing does, last
    assert _md5(path) == before


# --------------------------------------------------------------------------- #
# Quarantine of whole files (checkpoint 6)
# --------------------------------------------------------------------------- #
def test_unknown_layout_is_quarantined_and_fails_the_run(data_root, make_layout_a, make_unknown):
    good = make_layout_a(config.LANDING_DIR / "good.xlsx")
    make_unknown(config.LANDING_DIR / "mystery.xlsx")

    with pytest.raises(ingest_bronze.FileQuarantinedError):
        ingest_bronze.run()

    reason = config.QUARANTINE_DIR / "files" / "mystery.xlsx.reason.txt"
    assert not (config.LANDING_DIR / "mystery.xlsx").exists()
    assert (config.QUARANTINE_DIR / "files" / "mystery.xlsx").exists()
    assert config.REASON_UNKNOWN_LAYOUT in reason.read_text()
    # the good file was still ingested, and stays in landing for the next run
    assert [p.stem for p in _bronze_files()] == ["good"]
    assert good.exists()


def test_corrupted_file_is_quarantined_and_fails_the_run(data_root, make_corrupt):
    make_corrupt(config.LANDING_DIR / "broken.xlsx")

    with pytest.raises(ingest_bronze.FileQuarantinedError):
        ingest_bronze.run()

    reason = config.QUARANTINE_DIR / "files" / "broken.xlsx.reason.txt"
    assert config.REASON_UNREADABLE_FILE in reason.read_text()
    assert not (config.LANDING_DIR / "broken.xlsx").exists()
    assert _bronze_files() == []


def test_matching_layout_without_demand_column_is_quarantined(data_root, make_layout_b):
    make_layout_b(config.LANDING_DIR / "renamed.xlsx", demand_header="Load (MW)")

    with pytest.raises(ingest_bronze.FileQuarantinedError):
        ingest_bronze.run()

    reason = config.QUARANTINE_DIR / "files" / "renamed.xlsx.reason.txt"
    assert config.REASON_MISSING_COLUMNS in reason.read_text()
    assert _bronze_files() == []


# --------------------------------------------------------------------------- #
# Idempotency and edge cases
# --------------------------------------------------------------------------- #
def test_rerun_on_the_same_day_overwrites_instead_of_duplicating(data_root, make_layout_b):
    make_layout_b(config.LANDING_DIR / "b.xlsx")

    ingest_bronze.run()
    ingest_bronze.run()

    files = _bronze_files()
    assert len(files) == 1
    assert len(pd.read_parquet(files[0])) == 5 + 6


def test_empty_landing_is_not_an_error(data_root):
    summary = ingest_bronze.run()
    assert summary["ingested"] == []
    assert summary["quarantined"] == []


# --------------------------------------------------------------------------- #
# validate_bronze
# --------------------------------------------------------------------------- #
def test_validate_bronze_passes_on_a_good_run(data_root, make_layout_a, make_layout_b):
    make_layout_a(config.LANDING_DIR / "a.xlsx")
    make_layout_b(config.LANDING_DIR / "b.xlsx")
    quality.validate_bronze(ingest_bronze.run())   # must not raise


def test_validate_bronze_fails_when_rows_do_not_match(data_root, make_layout_b):
    make_layout_b(config.LANDING_DIR / "b.xlsx")
    summary = ingest_bronze.run()
    summary["ingested"][0]["expected_rows"] += 1   # pretend the Excel had one more row

    with pytest.raises(quality.QualityError):
        quality.validate_bronze(summary)


def test_validate_bronze_fails_when_nothing_was_ingested(data_root):
    with pytest.raises(quality.QualityError):
        quality.validate_bronze(ingest_bronze.run())