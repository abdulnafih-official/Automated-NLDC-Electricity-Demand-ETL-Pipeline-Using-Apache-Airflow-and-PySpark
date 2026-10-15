"""Shared helpers: logging, IST clock, file moves, Spark session factory."""

import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pyspark.sql import SparkSession

from src import config


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str) -> logging.Logger:
    """Console logger. Airflow captures stdout/stderr into the task log."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Time and ids (IST, no UTC conversion)
# --------------------------------------------------------------------------- #
def ist_now() -> datetime:
    return datetime.now(ZoneInfo(config.TIMEZONE))


def new_batch_id() -> str:
    """Unique per pipeline run, e.g. 20260715T093000_ab12cd34."""
    return f"{ist_now():%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
EXCEL_SUFFIXES = {".xlsx", ".xlsm"}


def list_landing_files() -> list[Path]:
    """Excel files waiting in landing. Anything here is unprocessed (plan section 7).

    Skips Excel lock files (~$name.xlsx) left by an open editor.
    """
    config.LANDING_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        p for p in config.LANDING_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in EXCEL_SUFFIXES and not p.name.startswith("~$")
    )


def move_file(src: Path, dest_dir: Path) -> Path:
    """Move a file into dest_dir. If the name is taken, add a timestamp suffix
    so nothing is ever overwritten."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        dest = dest_dir / f"{src.stem}__{ist_now():%Y%m%dT%H%M%S}{src.suffix}"
    shutil.move(str(src), str(dest))
    return dest


def quarantine_file(src: Path, reason: str) -> Path:
    """Move a whole bad file to quarantine and write a sidecar note with the reason."""
    dest = move_file(src, config.QUARANTINE_DIR / "files")
    note = dest.with_name(dest.name + ".reason.txt")
    note.write_text(f"reason: {reason}\nquarantined_at: {ist_now().isoformat()}\n")
    return dest


def archive_file(src: Path) -> Path:
    return move_file(src, config.ARCHIVE_DIR)


# --------------------------------------------------------------------------- #
# Spark
# --------------------------------------------------------------------------- #
def get_spark() -> SparkSession:
    """Local Spark session: IST session timezone, dynamic partition overwrite
    (so a rerun only replaces the partitions it writes), small shuffle."""
    tz = config.TIMEZONE
    return (
        SparkSession.builder
        .appName(config.SPARK_APP_NAME)
        .master(config.SPARK_MASTER)
        .config("spark.driver.memory", config.SPARK_DRIVER_MEMORY)
        .config("spark.sql.session.timeZone", tz)
        .config("spark.driver.extraJavaOptions", f"-Duser.timezone={tz}")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.sql.sources.partitionColumnTypeInference.enabled", "false")
        .config("spark.sql.shuffle.partitions", str(config.SPARK_SHUFFLE_PARTITIONS))
        .config("spark.ui.enabled", "false")
        .config("spark.jars.ivy", "/tmp/.ivy2")
        .getOrCreate()
    )