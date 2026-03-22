from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "credit_risk_dataset.csv"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_credit_risk_dataset(n_samples: int = 2400, random_state: int = 42) -> pd.DataFrame:
    """Generate a reproducible synthetic dataset for credit risk scoring."""
    rng = np.random.default_rng(random_state)

    age = rng.integers(21, 69, size=n_samples)
    annual_income = rng.lognormal(mean=10.45, sigma=0.42, size=n_samples)
    annual_income = np.clip(annual_income, 18000, 240000)

    employment_years = np.clip(rng.normal(loc=6.5, scale=4.0, size=n_samples), 0, 35)
    credit_history_years = np.clip(age - 18 - rng.normal(loc=5.0, scale=4.5, size=n_samples), 1, 40)
    loan_amount = np.clip(annual_income * rng.uniform(0.12, 0.75, size=n_samples), 2500, 90000)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], size=n_samples, p=[0.15, 0.2, 0.35, 0.2, 0.1])
    interest_rate = np.clip(rng.normal(loc=14.5, scale=4.2, size=n_samples), 5.5, 31.0)
    monthly_debt = np.clip((annual_income / 12) * rng.uniform(0.08, 0.62, size=n_samples), 150, 9000)
    debt_to_income = np.clip(monthly_debt / (annual_income / 12), 0.05, 0.95)
    credit_utilization = np.clip(rng.beta(2.0, 2.4, size=n_samples), 0.01, 0.99)
    late_payments_12m = rng.poisson(lam=1.1, size=n_samples)
    recent_credit_inquiries = rng.poisson(lam=1.5, size=n_samples)
    existing_loans = rng.integers(0, 6, size=n_samples)
    savings_balance = np.clip(
        annual_income * rng.uniform(0.02, 0.7, size=n_samples) - loan_amount * rng.uniform(0.0, 0.35, size=n_samples),
        0,
        180000,
    )

    home_ownership = rng.choice(
        ["rent", "mortgage", "own"],
        size=n_samples,
        p=[0.42, 0.38, 0.20],
    )
    loan_purpose = rng.choice(
        ["debt_consolidation", "home_improvement", "small_business", "education", "medical"],
        size=n_samples,
        p=[0.34, 0.18, 0.16, 0.15, 0.17],
    )
    employment_type = rng.choice(
        ["salaried", "self_employed", "contract"],
        size=n_samples,
        p=[0.68, 0.18, 0.14],
    )
    region = rng.choice(
        ["southeast", "south", "northeast", "midwest", "north"],
        size=n_samples,
        p=[0.33, 0.18, 0.2, 0.14, 0.15],
    )

    loan_to_income = loan_amount / annual_income

    risk_score = (
        -4.1
        + 4.2 * debt_to_income
        + 3.6 * credit_utilization
        + 0.35 * late_payments_12m
        + 0.18 * recent_credit_inquiries
        + 1.8 * loan_to_income
        + 0.15 * existing_loans
        - 0.05 * employment_years
        - 0.035 * credit_history_years
        - 0.000018 * annual_income
        - 0.000006 * savings_balance
        + np.where(home_ownership == "rent", 0.42, 0.0)
        + np.where(home_ownership == "mortgage", 0.12, -0.06)
        + np.where(loan_purpose == "small_business", 0.52, 0.0)
        + np.where(loan_purpose == "medical", 0.33, 0.0)
        + np.where(employment_type == "contract", 0.38, 0.0)
        + np.where(employment_type == "self_employed", 0.22, 0.0)
        + np.where(region == "north", 0.1, 0.0)
        + rng.normal(0, 0.12, size=n_samples)
    )

    probability_default = _sigmoid(risk_score)
    defaulted = rng.binomial(1, probability_default)

    dataset = pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income.round(2),
            "employment_years": employment_years.round(1),
            "credit_history_years": credit_history_years.round(1),
            "loan_amount": loan_amount.round(2),
            "loan_term_months": loan_term_months,
            "interest_rate": interest_rate.round(2),
            "monthly_debt": monthly_debt.round(2),
            "debt_to_income": debt_to_income.round(3),
            "credit_utilization": credit_utilization.round(3),
            "late_payments_12m": late_payments_12m,
            "recent_credit_inquiries": recent_credit_inquiries,
            "existing_loans": existing_loans,
            "savings_balance": savings_balance.round(2),
            "home_ownership": home_ownership,
            "loan_purpose": loan_purpose,
            "employment_type": employment_type,
            "region": region,
            "defaulted": defaulted,
        }
    )

    return dataset


def ensure_dataset(
    path: Path = DATA_PATH,
    n_samples: int = 2400,
    random_state: int = 42,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load the dataset from disk or generate it if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not refresh:
        return pd.read_csv(path)

    dataset = generate_credit_risk_dataset(n_samples=n_samples, random_state=random_state)
    dataset.to_csv(path, index=False)
    return dataset
