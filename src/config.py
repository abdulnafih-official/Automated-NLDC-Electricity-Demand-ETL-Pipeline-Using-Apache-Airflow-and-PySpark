"""Single source of truth for the NLDC ETL pipeline (plan section 8: no hardcoding in task code)."""

import os
import re
from pathlib import Path

# --------------------------------------------------------------------------- #
# Time zone: IST everywhere, never converted to UTC (plan section 2)
# --------------------------------------------------------------------------- #
TIMEZONE = "Asia/Kolkata"

# --------------------------------------------------------------------------- #
# Paths (inside the container; docker-compose mounts ./data here)
# --------------------------------------------------------------------------- #
DATA_ROOT = Path(os.environ.get("NLDC_DATA_ROOT", "/opt/airflow/data"))

LANDING_DIR = DATA_ROOT / "landing"
ARCHIVE_DIR = DATA_ROOT / "archive"
QUARANTINE_DIR = DATA_ROOT / "quarantine"
BRONZE_DIR = DATA_ROOT / "bronze"
SILVER_DIR = DATA_ROOT / "silver"
GOLD_DIR = DATA_ROOT / "gold"
HOURLY_MASTER_DIR = GOLD_DIR / "hourly_master"
DAILY_AGG_DIR = GOLD_DIR / "daily_agg"

# --------------------------------------------------------------------------- #
# Spark (local mode, modest memory for a 16 GB host)
# --------------------------------------------------------------------------- #
SPARK_APP_NAME = "nldc_etl"
SPARK_MASTER = "local[2]"
SPARK_DRIVER_MEMORY = "2g"
SPARK_SHUFFLE_PARTITIONS = 4  # data is tiny; the default of 200 just creates empty tasks

# --------------------------------------------------------------------------- #
# Source layouts (plan section 3)
# --------------------------------------------------------------------------- #
# Bronze column names: every source header is sanitised the same way so Spark
# can read them without backticks. "Demand (MW)" -> "Demand_MW",
# "NLDC_DEMAND|P" -> "NLDC_DEMAND_P", "Solar+Wind" -> "Solar_Wind".
def sanitize_column_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_")


SOURCE_LAYOUTS = {
    # Layout A: SCADA export, 5-minute grain, two header rows
    "A": {
        "match_sheets": {"Sheet1", "_osi_config"},  # all must be present to match
        "data_sheet": "Sheet1",
        "header_rows": 2,                # row 1 = tag ids, row 2 = display names
        "timestamp_col": "Time",         # bronze column name (row 1 label; row 2 is empty)
        "timestamp_format": None,        # real Excel datetimes, no string parsing
        "demand_col": "NLDC_DEMAND_P",   # tag SCADA/ANALOG/044MQ067/0
        "grain": "5min",
        "footer_markers": frozenset(),   # no footer rows seen
    },
    # Layout B: "Report", hourly grain, one header row, text values, footer stats rows
    "B": {
        "match_sheets": {"Report"},
        "data_sheet": "Report",
        "header_rows": 1,
        "timestamp_col": "Timestamp",
        "timestamp_format": "%d-%m-%Y %H:%M:%S",   # Spark pattern: dd-MM-yyyy HH:mm:ss
        "timestamp_format_spark": "dd-MM-yyyy HH:mm:ss",
        "demand_col": "Demand_MW",
        "grain": "hourly",
        # Values found in the timestamp column of the 6 footer rows
        "footer_markers": frozenset({"Minimum", "Maximum", "Average", "Sum", "Timestamp"}),
    },
}

# Layout A timestamps are already datetimes; Spark format only matters for B.
SOURCE_LAYOUTS["A"]["timestamp_format_spark"] = None

# --------------------------------------------------------------------------- #
# Bronze metadata columns
# --------------------------------------------------------------------------- #
BRONZE_META_COLUMNS = ["_source_file", "_source_layout", "_ingested_at", "_batch_id"]

# --------------------------------------------------------------------------- #
# Silver canonical schema (plan section 6)
# --------------------------------------------------------------------------- #
SILVER_COLUMNS = [
    "ts", "demand_mw", "source_grain", "_source_file", "_batch_id", "data_month",
]

# Quarantine reason codes
REASON_UNKNOWN_LAYOUT = "unknown_layout"
REASON_FOOTER_ROW = "footer_row"
REASON_UNPARSEABLE_TS = "unparseable_timestamp"
REASON_NON_NUMERIC_DEMAND = "non_numeric_demand"
REASON_NULL_DEMAND = "null_demand"
REASON_DEMAND_NONPOSITIVE = "demand_nonpositive"
REASON_DEMAND_OUT_OF_RANGE = "demand_out_of_range"
REASON_UNREADABLE_FILE = "unreadable_file"
REASON_MISSING_COLUMNS = "missing_columns"
REASON_DUPLICATE_SUPERSEDED = "duplicate_superseded"

# --------------------------------------------------------------------------- #
# Validation thresholds
# --------------------------------------------------------------------------- #
# Open decision #7 (provisional). Observed in the samples: 97,065 to 248,688 MW
# for valid readings. Zero and negative are always invalid, regardless of range.
DEMAND_VALID_MIN_MW = 50_000.0
DEMAND_VALID_MAX_MW = 350_000.0

# Open decision #3: an hour built from 5-minute data needs at least this many
# valid readings (out of 12), otherwise demand_mw is null for that hour.
MIN_READINGS_PER_HOUR = 9
EXPECTED_READINGS_PER_HOUR = {"5min": 12, "hourly": 1}

# Gold validation: fail the run if more than this share of hours has null demand
MAX_NULL_DEMAND_RATE = 0.05

# --------------------------------------------------------------------------- #
# Gold feature settings (plan section 6)
# --------------------------------------------------------------------------- #
LAG_HOURS = [1, 2, 3, 24, 48, 168]
ROLLING_WINDOWS_HOURS = {
    24: ["mean", "std", "min", "max"],
    168: ["mean", "std"],
}
SAME_HOUR_LOOKBACK_DAYS = 7

# Gold reads silver this many days before the target months so lag and rolling
# features are correct at the start of a month (168h window + 7-day same-hour window).
LOOKBACK_DAYS = 8

# India has four seasons in this project's definition (calendar month -> season)
SEASON_BY_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "summer", 4: "summer", 5: "summer",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post_monsoon", 11: "post_monsoon",
}