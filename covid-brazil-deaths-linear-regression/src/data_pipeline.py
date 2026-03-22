from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAILY_DATA_PATH = PROJECT_ROOT / "data" / "covid_brazil_daily_deaths_2024.csv"
OFFICIAL_SOURCE_URL = "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2024/INFLUD24-26-06-2025.csv"


def aggregate_daily_deaths_from_srag(raw_csv_path: Path, output_path: Path = DAILY_DATA_PATH) -> pd.DataFrame:
    """Aggregate official SRAG microdata into a national daily deaths series for COVID-19."""
    death_counter: Counter[str] = Counter()

    with raw_csv_path.open("r", encoding="latin-1", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            if row.get("CLASSI_FIN") != "5":
                continue
            if row.get("EVOLUCAO") != "2":
                continue

            death_date = row.get("DT_EVOLUCA", "").strip()
            if not death_date:
                continue

            try:
                parsed = datetime.strptime(death_date, "%Y-%m-%d").date()
            except ValueError:
                continue
            if parsed.year != 2024:
                continue

            death_counter[str(parsed)] += 1

    series = (
        pd.Series(death_counter, name="covid_deaths")
        .rename_axis("date")
        .sort_index()
        .reset_index()
    )
    series["date"] = pd.to_datetime(series["date"])

    full_range = pd.date_range(series["date"].min(), series["date"].max(), freq="D")
    frame = (
        series.set_index("date")
        .reindex(full_range, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )
    frame.columns = ["date", "covid_deaths"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def load_daily_dataset(path: Path = DAILY_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Daily dataset not found at {path}. Build it from the official SRAG source first."
        )
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["covid_deaths"] = frame["covid_deaths"].astype(int)
    return frame
