"""Shared pytest fixtures: an isolated data folder and builders for small sample Excel files.

The sample files are generated on the fly in each test's temp folder, so no binary
Excel files are committed and every test starts clean.
"""

from datetime import datetime, timedelta

import openpyxl
import pytest

from src import config

# Layout A header rows, copied from the real SCADA export (row 1 = tag ids, row 2 = display names)
A_ROW1 = [
    "Time", "SCADA/ANALOG/044MQ068/0", "SCADA/ANALOG/044MQ062/0", "SCADA/ANALOG/044MQ063/0",
    "SCADA/ANALOG/044MQ064/0", "SCADA/ANALOG/044MQ070/0", "SCADA/ANALOG/044MQ206/0",
    None, None, None,
    "SCADA/ANALOG/044MQ067/0", "SCADA/ANALOG/030K90IF/0", "SCADA/ANALOG/031E3193/0",
    "SCADA/ANALOG/034MT227/0", "SCADA/ANALOG/044MW005/0",
]
A_ROW2 = [
    None, "TOTAL_THM_ONLY|P", "TOTAL_NLDC_HYD|P", "TOTAL_NLDC_GAS|P", "TOTAL_NLDC_NPC|P",
    "ALL_INDIA_WIND|P", "ALL_IND_SOLAR|P", "Solar", "Solar+Wind", "Total",
    "NLDC_DEMAND|P", "INTRA_SYSTEM_INTERTIA|P", "ER_Total_Inertia|P", "SYS_IN|P", "SPARE 05",
]

# The six footer rows of the real Layout B file
B_FOOTER = [
    ["Minimum", "0", "0", "-2645.59", "0"],
    ["Timestamp", "16-01-2024 17:06:41", "16-01-2024 17:06:41", "30-03-2025 12:09:37", "18-05-2024 16:42:41"],
    ["Maximum", "291851.75", "20000.5", "30000.1", "259788.95"],
    ["Timestamp", "18-09-2024 09:02:25", "18-09-2024 09:02:25", "02-09-2024 18:52:22", "18-09-2024 09:02:25"],
    ["Average", "194009.98", "9000.1", "5000.2", "196299.98"],
    ["Sum", "2519025649.08", "100000.5", "202857959.8", "2548758902.69"],
]


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Point every pipeline path at an empty temp folder."""
    paths = {
        "LANDING_DIR": tmp_path / "landing",
        "ARCHIVE_DIR": tmp_path / "archive",
        "QUARANTINE_DIR": tmp_path / "quarantine",
        "BRONZE_DIR": tmp_path / "bronze",
        "SILVER_DIR": tmp_path / "silver",
        "GOLD_DIR": tmp_path / "gold",
        "HOURLY_MASTER_DIR": tmp_path / "gold" / "hourly_master",
        "DAILY_AGG_DIR": tmp_path / "gold" / "daily_agg",
    }
    for name, path in paths.items():
        monkeypatch.setattr(config, name, path)
    config.LANDING_DIR.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def make_layout_a():
    """Layout A (SCADA export): 5-minute rows, two header rows, junk sheets.
    zero_at = row indexes whose demand is 0.0 (a sensor dropout)."""

    def _make(path, start=datetime(2021, 9, 1), n_rows=36, zero_at=(5,)):
        wb = openpyxl.Workbook()
        cfg = wb.active
        cfg.title = "_osi_config"
        cfg["A1"] = 3
        cfg["A2"] = '{"r":300.0}'
        ws = wb.create_sheet("Sheet1")
        ws.append(A_ROW1)
        ws.append(A_ROW2)
        for i in range(n_rows):
            demand = 0.0 if i in zero_at else 150000.0 + i * 10
            ws.append([start + timedelta(minutes=5 * i)] + [1000.0 + i] * 9 + [demand] + [0.0] * 4)
        wb.create_sheet("Sheet2")
        wb.save(path)
        return path

    return _make


@pytest.fixture
def make_layout_b():
    """Layout B (Report): hourly rows, one header row, every value stored as text,
    plus the 6 footer rows."""

    def _make(path, start=datetime(2024, 1, 1), n_hours=5, footer=True, demand_header="Demand (MW)"):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Report"
        ws.append(["Timestamp", demand_header, "Wind (MW)", "Solar (MW)", "Total Generation (MW)"])
        for i in range(n_hours):
            ts = (start + timedelta(hours=i)).strftime("%d-%m-%Y %H:%M:%S")
            ws.append([ts, f"{141981.26 + i * 100:.2f}", "9545.07", "20.29", "142193.38"])
        if footer:
            for row in B_FOOTER:
                ws.append(row)
        wb.create_sheet("Sheet2")
        wb.create_sheet("Sheet3")
        wb.save(path)
        return path

    return _make


@pytest.fixture
def make_unknown():
    """A valid Excel file that matches neither layout."""

    def _make(path):
        wb = openpyxl.Workbook()
        wb.active.title = "Mystery"
        wb.active.append(["a", "b"])
        wb.save(path)
        return path

    return _make


@pytest.fixture
def make_corrupt():
    """A file with an .xlsx name that is not an Excel file."""

    def _make(path):
        path.write_text("this is not an excel file")
        return path

    return _make