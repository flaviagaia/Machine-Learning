from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customer_profiles.csv"


def generate_customer_dataset(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    segments = rng.choice(
        ["value_seekers", "loyal_midmarket", "premium_repeaters", "high_potential"],
        size=n_samples,
        p=[0.28, 0.30, 0.22, 0.20],
    )

    data = []
    for idx, segment in enumerate(segments, start=1):
        if segment == "value_seekers":
            annual_income = rng.normal(42000, 9000)
            purchase_frequency = rng.normal(3.5, 1.0)
            avg_ticket = rng.normal(90, 25)
            digital_engagement = rng.normal(45, 12)
            return_rate = rng.normal(8, 2.5)
            support_tickets = rng.normal(1.8, 0.8)
            loyalty_months = rng.normal(14, 6)
            discount_sensitivity = rng.normal(82, 8)
        elif segment == "loyal_midmarket":
            annual_income = rng.normal(72000, 12000)
            purchase_frequency = rng.normal(6.2, 1.2)
            avg_ticket = rng.normal(180, 35)
            digital_engagement = rng.normal(68, 10)
            return_rate = rng.normal(5, 1.8)
            support_tickets = rng.normal(1.2, 0.6)
            loyalty_months = rng.normal(34, 10)
            discount_sensitivity = rng.normal(54, 9)
        elif segment == "premium_repeaters":
            annual_income = rng.normal(118000, 18000)
            purchase_frequency = rng.normal(7.5, 1.5)
            avg_ticket = rng.normal(340, 55)
            digital_engagement = rng.normal(78, 8)
            return_rate = rng.normal(3.5, 1.3)
            support_tickets = rng.normal(0.9, 0.5)
            loyalty_months = rng.normal(42, 12)
            discount_sensitivity = rng.normal(28, 8)
        else:
            annual_income = rng.normal(88000, 15000)
            purchase_frequency = rng.normal(4.8, 1.1)
            avg_ticket = rng.normal(210, 40)
            digital_engagement = rng.normal(74, 9)
            return_rate = rng.normal(4.8, 1.6)
            support_tickets = rng.normal(1.0, 0.5)
            loyalty_months = rng.normal(18, 7)
            discount_sensitivity = rng.normal(46, 10)

        age = np.clip(rng.normal(38, 10), 20, 72)
        tenure_months = np.clip(loyalty_months + rng.normal(2, 4), 1, 72)
        web_visits_monthly = np.clip(digital_engagement / 7 + rng.normal(0, 2), 1, 40)
        app_sessions_weekly = np.clip(digital_engagement / 11 + rng.normal(0, 1.4), 1, 20)

        data.append(
            {
                "customer_id": f"C{idx:04d}",
                "age": round(float(age), 1),
                "annual_income": round(float(np.clip(annual_income, 18000, 220000)), 2),
                "purchase_frequency_monthly": round(float(np.clip(purchase_frequency, 0.5, 14)), 2),
                "average_ticket": round(float(np.clip(avg_ticket, 20, 650)), 2),
                "digital_engagement_score": round(float(np.clip(digital_engagement, 5, 100)), 2),
                "return_rate_pct": round(float(np.clip(return_rate, 0, 18)), 2),
                "support_tickets_quarter": round(float(np.clip(support_tickets, 0, 8)), 2),
                "loyalty_months": round(float(np.clip(loyalty_months, 1, 72)), 1),
                "discount_sensitivity_score": round(float(np.clip(discount_sensitivity, 5, 100)), 2),
                "web_visits_monthly": round(float(web_visits_monthly), 2),
                "app_sessions_weekly": round(float(app_sessions_weekly), 2),
                "tenure_months": round(float(tenure_months), 1),
                "segment_label": segment,
            }
        )

    return pd.DataFrame(data)


def ensure_dataset(path: Path = DATA_PATH, n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_csv(path)

    dataset = generate_customer_dataset(n_samples=n_samples, random_state=random_state)
    dataset.to_csv(path, index=False)
    return dataset
