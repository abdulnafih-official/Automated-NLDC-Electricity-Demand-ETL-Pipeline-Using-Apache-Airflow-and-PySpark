"""nldc_daily_etl: the single daily DAG (plan section 7).

landing Excel -> bronze -> silver -> gold (hourly_master + daily_agg) -> archive.
Task order is linear. A failed validation stops the DAG, and Excel files are archived
last, so a failed run can be retried with the files still in landing.
"""

from datetime import timedelta

import pendulum
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from airflow.sensors.python import PythonSensor


# src modules are imported inside the task functions, not at the top of this file,
# so Airflow can parse the DAG quickly without loading pyspark and pandas.

def _landing_has_files() -> bool:
    from src.utils import list_landing_files

    return len(list_landing_files()) > 0


def _gate(check, summary: dict) -> None:
    """Run a validation gate. A failed gate is a hard failure: retrying the same bad data
    would fail the same way, so no retries."""
    from src.quality import QualityError

    try:
        check(summary)
    except QualityError as exc:
        raise AirflowFailException(str(exc)) from exc


@dag(
    dag_id="nldc_daily_etl",
    schedule="0 6 * * *",  # 06:00 IST every day
    start_date=pendulum.datetime(2025, 1, 1, tz="Asia/Kolkata"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["nldc", "etl"],
    doc_md=__doc__,
)
def nldc_daily_etl():

    # 1. wait briefly for files; if landing stays empty the run is skipped, not failed
    check_landing = PythonSensor(
        task_id="check_landing",
        python_callable=_landing_has_files,
        mode="reschedule",
        poke_interval=30,
        timeout=120,
        soft_fail=True,
    )

    # 2. landing Excel -> bronze Parquet
    @task
    def bronze() -> dict:
        from src import ingest_bronze

        try:
            return ingest_bronze.run()
        except ingest_bronze.FileQuarantinedError as exc:
            raise AirflowFailException(str(exc)) from exc  # quarantined file fails the run

    # 3.
    @task
    def validate_bronze(bronze_summary: dict) -> None:
        from src import quality

        _gate(quality.validate_bronze, bronze_summary)

    # 4. bronze -> silver (rejected rows go to quarantine/rows)
    @task
    def silver(bronze_summary: dict) -> dict:
        from src import clean_silver

        return clean_silver.run(bronze_summary["batch_id"])

    # 5.
    @task
    def validate_silver(silver_summary: dict) -> None:
        from src import quality

        _gate(quality.validate_silver, silver_summary)

    # 6. silver -> hourly resample -> features -> hourly_master + daily_agg
    @task
    def gold(silver_summary: dict) -> dict:
        from src import build_gold

        return build_gold.run(silver_summary["months_touched"])

    # 7.
    @task
    def validate_gold(gold_summary: dict) -> None:
        from src import quality

        _gate(quality.validate_gold, gold_summary)

    # 8. last step: move the processed Excel files out of landing
    @task
    def archive_landing(bronze_summary: dict) -> None:
        from src import config
        from src.utils import archive_file, get_logger

        log = get_logger("archive_landing")
        for item in bronze_summary["ingested"]:
            path = config.LANDING_DIR / item["file"]
            if path.exists():
                log.info("archived %s -> %s", item["file"], archive_file(path))

    b = bronze()
    vb = validate_bronze(b)
    s = silver(b)
    vs = validate_silver(s)
    g = gold(s)
    vg = validate_gold(g)
    a = archive_landing(b)

    check_landing >> b >> vb >> s >> vs >> g >> vg >> a


nldc_daily_etl()