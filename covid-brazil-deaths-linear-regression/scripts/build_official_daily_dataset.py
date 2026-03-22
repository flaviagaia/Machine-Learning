from __future__ import annotations

import csv
import io
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "covid_brazil_daily_deaths_2024.csv"


def main() -> None:
    counter: Counter[str] = Counter()
    text_stream = io.TextIOWrapper(sys.stdin.buffer, encoding="latin-1", newline="")
    reader = csv.DictReader(text_stream, delimiter=";")

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

        counter[str(parsed)] += 1

    series = pd.Series(counter, name="covid_deaths").rename_axis("date").sort_index().reset_index()
    series["date"] = pd.to_datetime(series["date"])

    full_range = pd.date_range(series["date"].min(), series["date"].max(), freq="D")
    frame = series.set_index("date").reindex(full_range, fill_value=0).rename_axis("date").reset_index()
    frame.columns = ["date", "covid_deaths"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)

    print(OUTPUT_PATH)
    print(f"rows={len(frame)}")
    print(f"total_deaths={int(frame['covid_deaths'].sum())}")


if __name__ == "__main__":
    main()
